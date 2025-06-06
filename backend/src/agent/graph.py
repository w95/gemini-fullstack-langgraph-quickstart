import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    model_name = configurable.query_generator_model
    llm: BaseChatModel

    if model_name.startswith(("gpt-", "text-davinci-")):
        openai_api_key = configurable.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set for OpenAI model")
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=1.0,
            max_retries=2,
        )
    else:  # Assume Gemini model
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set for Gemini model")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=1.0,
            max_retries=2,
        )

    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    search_model_name = configurable.query_generator_model # This is the model for summarizing search results

    # For the actual search operation, we always use the google.generativeai Client
    # as it provides the google_search tool and grounding metadata.
    gemini_api_key_for_search = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key_for_search:
        raise ValueError("GEMINI_API_KEY is not set, required for web_research's search tool functionality.")

    # We use the query_generator_model for the search tool itself, as it might be tuned for search.
    # This client is only for the search part.
    genai_search_client = Client(api_key=gemini_api_key_for_search)

    search_prompt_for_tool = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"], # This is the individual query from the list
    )

    # Perform the search using Gemini (even if summarization is by OpenAI)
    search_response = genai_search_client.models.generate_content(
        model=configurable.query_generator_model, # Use the query_generator_model for the search tool
        contents=search_prompt_for_tool,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0, # Low temperature for factual search
        },
    )

    # Process search results (URL resolution and citation extraction)
    resolved_urls = resolve_urls(
        search_response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    citations = get_citations(search_response, resolved_urls)

    # Now, generate the actual textual content using the selected LLM
    # The content from the search tool is in search_response.text
    search_result_text = search_response.text

    llm_for_summarization: BaseChatModel
    if search_model_name.startswith(("gpt-", "text-davinci-")):
        openai_api_key = configurable.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set for OpenAI model selected for web research summarization.")
        llm_for_summarization = ChatOpenAI(
            model=search_model_name,
            openai_api_key=openai_api_key,
            temperature=0.7, # Allow some creativity in summarization
            max_retries=2,
        )
        # We need to craft a new prompt for OpenAI to summarize/answer based on search_result_text
        # The original web_searcher_instructions was for the Gemini tool.
        # For now, we will directly use the search_result_text and then add citations.
        # A more sophisticated approach might involve a prompt like:
        # "Based on the following search results: {search_result_text}\n\nPlease answer the query: {state['search_query']}"
        # However, the existing web_searcher_instructions already asked the model to "generate a comprehensive summary".
        # So, we'll assume the search_result_text is the summary, and then let OpenAI refine it if needed,
        # or simply use it as is if the model is Gemini.

        # For OpenAI, let's re-run the web_searcher_instructions with the context from Gemini search
        # This is a bit indirect. A better way would be to have a dedicated summarization prompt.
        prompt_for_openai_summarization = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=state["search_query"] + f"\n\nUse the following information to answer:\n{search_result_text}"
        )
        generated_text_response = llm_for_summarization.invoke(prompt_for_openai_summarization)
        generated_text = generated_text_response.content

    else: # Gemini model was used for search_model_name
        # The search_response.text already contains the generated content from Gemini
        generated_text = search_result_text

    modified_text = insert_citation_markers(generated_text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]], # Keep track of the original query for this result
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    # The line below had configurable.reasoning_model, it should be reflection_model
    # reasoning_model = state.get("reasoning_model") or configurable.reasoning_model
    model_name = configurable.reflection_model


    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    llm: BaseChatModel
    if model_name.startswith(("gpt-", "text-davinci-")):
        openai_api_key = configurable.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set for OpenAI model")
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=1.0,
            max_retries=2,
        )
    else:  # Assume Gemini model
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set for Gemini model")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=1.0,
            max_retries=2,
        )

    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    # The line below had configurable.reasoning_model, it should be answer_model
    # reasoning_model = state.get("reasoning_model") or configurable.reasoning_model
    model_name = configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm: BaseChatModel
    if model_name.startswith(("gpt-", "text-davinci-")):
        openai_api_key = configurable.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set for OpenAI model")
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=0,
            max_retries=2,
        )
    else:  # Assume Gemini model
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set for Gemini model")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0,
            max_retries=2,
        )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
