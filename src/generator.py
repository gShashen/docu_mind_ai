import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

def generate_awnswer(query, retrived_chunks):
    chunks = []

    for chunk in retrived_chunks:
        chunks.append(chunk["chunk"])
    
    text = " ".join(chunks)


    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
    If the answer is in the context, state it directly with the exact figure or statistic.
    If the answer is genuinely not in the context, say so.

    Context:
    {text}

    Question:
    {query}"""

    message = client.messages.create(
        model = "claude-haiku-4-5",
        max_tokens= 1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ] 
    )

    return message.content[0].text

if __name__ == "__main__":
    if api_key:
        print("OKK")
    else:
        print("NOO")