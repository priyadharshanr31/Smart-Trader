from dotenv import load_dotenv
load_dotenv()

from core.data_manager import DataManager
from core.llm import LCTraderLLM
from agents.short_term_agent import ShortTermAgent
from agents.mid_term_agent import MidTermAgent
from agents.long_term_agent import LongTermAgent

def debug_symbol(sym: str):
    print("\n" + "=" * 80)
    print(f"DEBUG FOR {sym}")
    print("=" * 80)

    dm = DataManager()
    snap = dm.layered_snapshot_crypto(sym)

    # Show shapes of the dataframes the agents see
    for layer in ["short_term", "mid_term", "long_term"]:
        df = snap.get(layer)
        print(f"{layer}:",
              "None" if df is None else f"shape={df.shape}, cols={list(df.columns)}")
        if df is not None:
            print(df.tail(3))  # last 3 rows just to eyeball

    llm = LCTraderLLM()

    st = ShortTermAgent("ShortTerm", llm, {})
    mt = MidTermAgent("MidTerm", llm, {})
    lt = LongTermAgent("LongTerm", llm, {}, semantic_memory=None)

    print("\n--- AGENT VOTES ---")
    for label, agent in [
        ("Short", st),
        ("Mid", mt),
        ("Long", lt),
    ]:
        decision, conf, raw = agent.vote(snap)
        print(f"{label} -> decision={decision}, conf={conf}")
        print(f"{label} raw (first 400 chars): {str(raw)[:400]}")
        print("-" * 40)

def main():
    for sym in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        debug_symbol(sym)

if __name__ == "__main__":
    main()
