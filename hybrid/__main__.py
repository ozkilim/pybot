"""Chat with the cascade in a terminal.

    python -m hybrid                       # chat; each reply is tagged [python] or [model]
    python -m hybrid -p "why is the sky blue?"
    python -m hybrid --budget 0.01         # let the model take about 1% of everyday turns
"""
import argparse
import sys

from . import Cascade


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybrid", description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--prompt", help="reply to this one message and exit")
    parser.add_argument("-b", "--budget", default=None,
                        help="target share of turns the model may answer, e.g. 0.01, 0.10, 0.30")
    args = parser.parse_args(argv)
    bot = Cascade(budget=args.budget)

    def show(out):
        who = "model:%s" % out["model"] if out["tier"] == "llm" else "python:%s" % out["module"]
        print("%s\n  [%s | %.0f ms | code %.0f%% vs model %.0f%%]"
              % (out["reply"], who, out["code_ms"] + out["llm_ms"], 100 * out["p_code"], 100 * out["p_llm"]))

    memory = {}
    if args.prompt:
        show(bot.turn(args.prompt, memory))
        return 0
    print("PyBot Hybrid: Python first, %s when the code should not answer. /exit quits." % bot.llm.name)
    while True:
        try:
            text = input("\nuser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in ("/exit", "/quit"):
            break
        if text == "/reset":
            memory = {}
            continue
        if text:
            show(bot.turn(text, memory))
    stats = bot.stats
    if stats["turns"]:
        print("%d turns: %d by Python, %d by the model." % (stats["turns"], stats["code"], stats["llm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
