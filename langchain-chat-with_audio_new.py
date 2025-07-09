from datetime import datetime
import tempfile
import uuid
from IPython.display import Image
from typing import List, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import get_buffer_string
import tiktoken
from langchain_core.documents import Document
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
import os
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from website_scraper_new import scrape_website
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import RecursiveUrlLoader

import gputil

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    beta_use_converse_api=True,
    temperature=0.0,
    model_kwargs={
        "top_p": 0.1
    }
)

tokenizer = tiktoken.encoding_for_model("gpt-4o")

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

recall_vector_store = Chroma(collection_name="site-recall", embedding_function=embeddings,
                             persist_directory="./recall_chroma_db")
# recall_vector_store.delete_collection()
persist_directory = "./dev_portal_chroma_db_new"
vector_store = Chroma(collection_name="gpn-devportal_new", embedding_function=embeddings,
                      persist_directory=persist_directory)


# recall_vector_store = Chroma(collection_name="site-recall-accounts", embedding_function=embeddings,
#                              persist_directory="./recall_accounts_chroma_db")
# persist_directory = "./accounts_chroma_db_new"
# vector_store = Chroma(collection_name="accounts_new", embedding_function=embeddings,
#                       persist_directory=persist_directory)


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "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"
# -----------------------------------------------------------------------------------------

# start_url = "https://help.globalpaymentsintegrated.com/merchantportal/accounts/"
# mydocs = asyncio.run(scrape_website(start_url))
# print("Starting character splitting.")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# print("Character splitting completed.")
# all_splits = text_splitter.split_documents(mydocs)
# print("Adding documents in vector store.")
# _ = vector_store.add_documents(documents=all_splits)
# print("Adding documents in vector store completed")


# --------------------------------------------------------------------------------------------
# print("Vector store data  ", vector_store.get())


class State(MessagesState):
    recall_memories: List[str]
    docs_content: str


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    print("............... save_recall_memory ......................")
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""

    print("............... save_recall_memory ......................")

    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3
    )

    return []

    # return [document.page_content for document in documents]


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve and rerank information related to a query using Cohere's Reranker on AWS Bedrock."""
    print("............... Retrieve ......................")
    # Retrieve top-k documents from the vector store
    retrieved_docs = vector_store.similarity_search(query, k=50)

    print(f"##################### Retrieved  {len(retrieved_docs)} from the vector database")

    # Prepare documents for reranking
    documents = [doc.page_content for doc in retrieved_docs]

    # Initialize AWS Bedrock client
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

    # Construct payload for Cohere reranker on Bedrock
    payload = {
        "query": query,
        "documents": documents,
        "top_n": 10,
        "api_version": 2
    }

    # Invoke the Cohere reranker model on AWS Bedrock
    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId="cohere.rerank-v3-5:0"
    )

    # Parse response
    response_body = json.loads(response["body"].read())
    rerank_results = response_body["results"]

    # Sort retrieved documents by rerank score
    reranked_docs = [retrieved_docs[result["index"]] for result in rerank_results]

    print(f"################  Retrieved  {len(reranked_docs)} from the raranker")

    # Serialize results
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in reranked_docs
    )

    return serialized, reranked_docs


# def retrieve(query: str):
#     """Retrieve information related to a query from the vector store"""
#     retrieved_docs = vector_store.similarity_search(query, k=50)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs


tools = [retrieve, save_recall_memory, search_recall_memories]


def generate(state: State):
    print("............... generate ......................")
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = ("You are an assistant for question-answering tasks. "
                              "Use the following pieces of retrieved context to answer "
                              f"Context : {docs_content})"
                              "Important :  Restrict yourself to the information provided in context for answering  do not use your background knowledge or any industry research."
                              "Provide Document references at the end so that user can navigate to the documents and do more analysis. Make sure references are clickable"
                              "If you do not have enough details to give a full answer, Respond only with 'I do not have the required information to provide details on this'"
                              "\n\n")

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
           or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    print("----------------------------------------")
    print(prompt)
    print("----------------------------------------")
    prediction = llm.invoke(prompt)

    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.
    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.
    Returns:
        State: The updated state with loaded memories.
    """

    print("............... load_memories ......................")

    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {"recall_memories": recall_memories}


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: State):
    """Generate tool call for retrieval or respond."""
    print("............... query_or_respond ......................")
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node(load_memories)
graph_builder.add_node(query_or_respond)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node(generate)
graph_builder.add_edge(START, "load_memories")
graph_builder.add_edge("load_memories", "query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
graph = graph_builder.compile()
print(graph.get_graph().draw_mermaid())


# image = Image(graph.get_graph().draw_mermaid_png())
# with open(r"myGraph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())


def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print("pretty_print_stream_chunk     ", node.title(), updates)

        print("\n")


###############################################################################################
result = ""
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    global result
    data = request.get_json(force=True)
    config = {"configurable": {"thread_id": "rishi154", "user_id": "rishi"}, "recursion_limit": 50}
    input_message = data["message"]

    print(f"Input Message  :   {input_message}")
    response = ""
    for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}],
             "recall_memories": ""},
            stream_mode="values",
            config=config,
            debug=True
    ):
        # pretty_print_stream_chunk(step)
        response = step["messages"][-1].content
        result = {
            "result": response
        }

    return json.dumps(result)


_project_id = "steam-collector-339717"
_location = "us-central1"
_model_name = "gemini-1.5-flash-002"
_model = gputil.init_vertexai_with_model(_project_id, _location, _model_name)


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        logger.info("Received transcription request")

        if _model is None:
            logger.error("Model not loaded properly")
            return jsonify({"error": "Speech recognition model not available"}), 500

        if 'audio' not in request.files:
            logger.warning("No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        logger.info(f"Received file: {audio_file.filename}")

        if not audio_file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Save uploaded file temporarily
        temp_filename = None
        try:
            # Create unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(audio_file.filename)[1] or '.webm'
            temp_filename = os.path.join(tempfile.gettempdir(), f"whisper_{timestamp}{ext}")

            # Save the file
            audio_file.save(temp_filename)
            logger.info(f"Audio saved to temporary file: {temp_filename}")

            # Check if file exists and has content
            if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                logger.error("Temp file is empty or doesn't exist")
                return jsonify({"error": "Failed to save audio file"}), 500

            # Transcribe with Faster Whisper
            logger.info("Starting transcription")

            # Run the transcription
            response = gputil.transcribe_audio(temp_filename, _model)

            return jsonify({
                "success": True,
                "transcription": response,
                # "language": info.language,
                # "language_probability": round(info.language_probability, 3)
            })

        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            return jsonify({"error": f"Transcription error: {str(e)}"}), 500

        finally:
            # Clean up the temporary file
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                    logger.info("Temporary file removed")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return jsonify({"error": "Server error: " + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
