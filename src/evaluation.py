import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.rag_pipeline import load_vector_db, retrieve_documents, build_context, generate_answer

load_dotenv()

# Setup LLM for Ragas (Usually needs OpenAI for reliable scores)
eval_llm = ChatOpenAI(model="gpt-4o-mini")

def run_evaluation():
    print("🚀 Starting RAGAS Evaluation...")
    vectordb = load_vector_db()
    
    # Test Queries (Clinical Scenarios)
    test_queries = [
        "What are the diagnostic criteria for T2DM in symptomatic individuals?",
        "How is hypertension defined in Malaysian guidelines?",
        "What are the treatment targets for a patient with diabetes and hypertension?",
        "When should an OGTT be performed?"
    ]
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] # In a real scenario, you'd provide golden answers here
    }

    # Generate RAG results
    for query in test_queries:
        print(f"📝 Testing: {query}")
        docs = retrieve_documents(query, vectordb)
        context = [doc.page_content for doc in docs]
        answer = generate_answer(query, build_context(docs))
        
        data["question"].append(query)
        data["answer"].append(answer)
        data["contexts"].append(context)
        # Mocking ground truth for this demo - normally you'd have curated these
        data["ground_truth"].append("") 

    # Convert to dataset
    dataset = Dataset.from_dict(data)

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        llm=eval_llm
    )

    print("\n📊 RAGAS Evaluation Results:")
    print("-" * 30)
    df = result.to_pandas()
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])
    
    # Save results
    df.to_csv("ragas_report.csv", index=False)
    print("\n✅ Detailed report saved to ragas_report.csv")

if __name__ == "__main__":
    run_evaluation()