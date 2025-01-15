from langchain_aws import ChatBedrockConverse
import boto3

from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.CRITICAL)

dynamodb = boto3.resource("dynamodb")
client = boto3.client('sts')
def get_iam_user_id():    
    # Get the caller identity
    identity = client.get_caller_identity()
    
    # Extract and return the User ID
    user_id = identity['UserId']
    return user_id

# Get UserId for sessionId
user_id = get_iam_user_id()

model = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens=2048,
    temperature=0.0,
    top_p=1,
    stop_sequences=["\n\nHuman"],
    verbose=True
)

try:
    # Create the DynamoDB table.
    table = dynamodb.create_table(
        TableName="SessionTable",
        KeySchema=[{"AttributeName": "SessionId", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    # Wait until the table exists.
    table.meta.client.get_waiter("table_exists").wait(TableName="SessionTable")

    # Print out some data about the table.
    print(table.item_count)
except Exception as e:
    print(e)


# Initialize the DynamoDB chat message history
history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="0")

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create output parser to simplify the output
output_parser = StrOutputParser()

# Combine the prompt with the Bedrock LLM
chain = prompt | model| output_parser

# Integrate with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# Invoke the chain with a session-specific configuration
config = {"configurable": {"session_id": user_id}}



response = chain_with_history.invoke({"question": "What's my name?"}, config=config)
print(response)