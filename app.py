import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

es_client = Elasticsearch("http://localhost:9200")

# Define the index name
index_name = "conversations"

if es_client.indices.exists(index=index_name):
    print("Index exists")
else:
    # Define the index mapping
    index_mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "Context": {"type": "text"},
                "Response": {"type": "text"},
            }
        },
    }

    # Create the index with the mapping
    es_client.indices.create(index=index_name, body=index_mapping)


def build_prompt_ollama(query, search_re):
    # Define the system-level instructions
    system_prompt = """
You are a highly empathetic and supportive mental health counseling assistant. Your task is to assist the PATIENT by providing detailed, thoughtful, and compassionate responses solely based on the CONTEXT from the Mental Health Counseling Conversations database.

### Guidelines:
1. Always use information explicitly found in the CONTEXT when responding to the PATIENT.
   - Even if the CONTEXT isn't directly aligned, synthesize any insights that might be relevant or helpful to the patient's situation.
2. If the CONTEXT contains multiple examples, leverage as many as necessary to craft a comprehensive, empathetic response.
3. Write your response as a single, cohesive paragraph in the first-person singular perspective (e.g., "I understand that...").
4. Maintain a supportive and conversational tone — be understanding, empathetic, and encouraging.
5. Provide thoughtful, comprehensive answers. Avoid short or superficial responses.  
6. Use the response "I'm sorry, I don't have enough information to answer that right now." ONLY if the CONTEXT truly lacks sufficient information.
   - If the CONTEXT contains any helpful insights, you must provide a response.

### Example Response Format:
Response: I understand that suicidal thoughts can be incredibly overwhelming, and I'm truly sorry you're going through this. Based on similar situations in the database, seeking support from a trusted mental health professional can be very important during these difficult moments. Talking to someone you trust, such as a friend or family member, might also help reduce some of the pressure you're feeling. You don't have to face this alone, and there are people who care and can help you.
    """.strip()

    # Build the context content dynamically
    context = "\n".join(
        f"Database Patient: {doc.get('_source', {}).get('Context', 'N/A')}\n"
        f"Database Response: {doc.get('_source', {}).get('Response', 'N/A')}"
        for doc in search_re
    )

    # Define user-specific content
    user_prompt = f"""
PATIENT: {query}

CONTEXT from Mental Health Counseling Conversations database:
{context}
""".strip()

    return f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{user_prompt}\n<|eot_id|>"


def llm_ollama(prompt):
    # Fixed: Changed client_ollama to client
    response = client.chat.completions.create(
        model="llama3.2", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def rag_ollama(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["Context^2", "Response"],
                        "type": "most_fields",
                    }
                }
            }
        },
    }

    response = es_client.search(index="conversations", body=search_query)
    search_re = response["hits"]["hits"]

    prompt = build_prompt_ollama(query, search_re)
    answer = llm_ollama(prompt)
    print(prompt)
    print("")
    print("-" * 40)
    print("")
    return answer


def main():
    st.title("RAG Function Invocation")

    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            output = rag_ollama(user_input)
            st.success("Completed!")
            st.write(output)


if __name__ == "__main__":
    main()
