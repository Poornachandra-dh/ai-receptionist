import os
import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

load_dotenv(dotenv_path=root_dir / ".env")

def benchmark_rag_latency():
    print("=" * 60)
    print("SUB-500MS ARCHITECTURE LATENCY BENCHMARK")
    print("=" * 60)

    from rag.ingest import DocumentIngester
    from rag.retriever import RAGRetriever

    data_dir = "./data/bench_rag"
    sample_file = "./data/sample_clinic_faq.md"

    print("\n[1] Preparing Benchmark FAISS Index...")
    ingester = DocumentIngester(data_dir=data_dir)
    ingester.ingest_file(sample_file)

    retriever = RAGRetriever(data_dir=data_dir)
    query = "What happens if I need to cancel my appointment?"

    # Warm-up run
    retriever.retrieve(query)

    # Benchmark 10 iterations
    timings = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = retriever.retrieve(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)

    avg_ms = sum(timings) / len(timings)
    min_ms = min(timings)
    print(f" -> Local RAG Retrieval (FastEmbed ONNX + FAISS):")
    print(f"    Average: {avg_ms:.2f} ms | Fastest: {min_ms:.2f} ms")
    print(f"    Confidence Score: {res['max_score']:.4f}")

    if avg_ms < 25.0:
        print("    [PASS] Vector search is well within sub-500ms budget (<25ms)!")
    else:
        print("    [WARN] Retrieval took > 25ms, verify ONNX CPU settings.")

    # Cleanup benchmark dir
    import shutil
    if Path(data_dir).exists():
        shutil.rmtree(data_dir)

def benchmark_groq_ttft():
    print("\n[2] Benchmarking Groq LPU Time-To-First-Token (TTFT)...")
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or "your_groq_api_key" in groq_key:
        print("    [SKIP] GROQ_API_KEY not set in .env. Skipping cloud LLM benchmark.")
        return

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in three words."}],
            stream=True,
            max_tokens=20
        )
        
        first_token_time = None
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta and first_token_time is None:
                first_token_time = (time.perf_counter() - t0) * 1000.0
                break
                
        if first_token_time:
            print(f"    Model: {model}")
            print(f"    Groq TTFT (Time-To-First-Token): {first_token_time:.2f} ms")
            if first_token_time < 200.0:
                print("    [PASS] Groq TTFT is blisteringly fast (<200ms)!")
    except Exception as e:
        print(f"    [WARN] Groq benchmark encountered error: {e}")

if __name__ == "__main__":
    benchmark_rag_latency()
    benchmark_groq_ttft()
    print("\n" + "=" * 60)
