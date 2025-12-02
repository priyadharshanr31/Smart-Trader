from dotenv import load_dotenv
load_dotenv()  # make sure .env is loaded

from core.llm import LCTraderLLM

def main():
    llm = LCTraderLLM()  # uses GEMINI_API_KEY + GEMINI_MODEL from env
    system_msg = "You are a trading test LLM. Return JSON only."
    user_tmpl = (
        "Ticker: {ticker}\n"
        "Task: Decide BUY/SELL/HOLD with confidence in [0,1].\n"
        "Dummy data:\n{table}\n"
        "Return ONLY JSON."
    )
    variables = {
        "ticker": "TEST",
        "table": "close\n100\n101\n102\n103\n"
    }

    decision, conf, raw = llm.vote_structured(
        system_msg=system_msg,
        user_template=user_tmpl,
        variables=variables,
    )

    print("Decision:", decision)
    print("Confidence:", conf)
    print("Raw LLM output:\n", raw)

if __name__ == "__main__":
    main()
