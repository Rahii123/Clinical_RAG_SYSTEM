import os
import json
from typing import List
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall
)
from datasets import Dataset
from src.rag_pipeline import load_vector_db, retrieve_documents, build_context, generate_answer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# We need an LLM for Ragas evaluation (typically GPT-4 or similar)
# If OPENAI_API_KEY is present, we use it for evaluation.
eval_llm = ChatOpenAI(model="gpt-4o")

def run_production_eval():
    print("🧪 Starting Clinical RAG Evaluation (RAGAS)...")
    
    vectordb = load_vector_db()
    
    # Define test cases (Gold Standard)
    test_queries = [
        {
            "question": "How to diagnose T2DM in a symptomatic individual?",
            "ground_truth": "A single abnormal VPG (>= 7.0 mmol/L) or HbA1c (>= 6.3%) or RPG (>= 11.1 mmol/L) is sufficient."
        },
        {
            "question": "What are the BP targets for a patient with T2DM?",
            "ground_truth": "The target BP for patients with T2DM is < 140/90 mmHg."
        },
        {
            "question": "Is HbA1c used for diagnosis in patients with renal failure?",
            "ground_truth": "No, HbA1c is not reliable in conditions with rapid red cell turnover like renal failure."
        }
    ]
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for test in test_queries:
        print(f"  Evaluating: {test['question']}")
        
        # 1. Pipeline Run
        docs = retrieve_documents(test['question'], vectordb)
        context_str = build_context(docs)
        answer = generate_answer(test['question'], context_str)
        
        # 2. Collect Data
        data["question"].append(test['question'])
        data["answer"].append(answer.split("---")[0].strip())
        data["contexts"].append([d.page_content for d in docs])
        data["ground_truth"].append(test['ground_truth'])
        
    # Create Dataset
    dataset = Dataset.from_dict(data)
    
    # 3. Evaluate
    print("  Calculating Ragas scores...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=eval_llm
    )
    
    # 4. Report
    print("\n📊 RAGAS EVALUATION REPORT")
    print("=" * 30)
    for metric, score in result.items():
        print(f"{metric.capitalize()}: {score:.4f}")
    print("=" * 30)
    
    return result

if __name__ == "__main__":
    try:
        run_production_eval()
    except Exception as e:
        print(f"❌ Eval failed: {e}")
        print("Note: Ragas evaluation requires valid OPENAI_API_KEY for the judge LLM.")
