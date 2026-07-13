"""
dspy_qa_bootstrap.py
====================

A minimal, end-to-end example of optimizing a factual Q&A program with DSPy's
`BootstrapFewShot` optimizer.

The point of this file is to show the *shape* of a real DSPy optimization loop:
you configure a live LM backend, declare a signature, wrap it in a module,
supply labeled examples + a metric, and let the optimizer discover good few-shot
demonstrations for you. It talks to a real API (Anthropic here), so you would
run it exactly like production code.

The five moving parts every DSPy optimization has:
    1. An LM backend            -> configure_lm()
    2. A Signature (the task)   -> FactualQA
    3. A Module (the program)   -> QAProgram
    4. Data + a Metric          -> build_dataset(), answer_exact_match()
    5. An Optimizer             -> BootstrapFewShot in main()

Run it:
    pip install -U dspy
    export ANTHROPIC_API_KEY="sk-ant-..."
    python dspy_qa_bootstrap.py
"""

import os

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate


# ---------------------------------------------------------------------------
# 1. Configure the language model backend
# ---------------------------------------------------------------------------
def configure_lm() -> None:
    """Point DSPy at a live LM.

    DSPy routes all model calls through LiteLLM, so the model string is
    "<provider>/<model>". The API key is read from the environment rather than
    hard-coded, which is how you'd wire this in a real service.
    """
    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.0,       # deterministic answers make eval reproducible
        max_tokens=512,
    )
    # dspy.configure sets the default LM for every module in this process.
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# 2. Declare the task as a Signature
# ---------------------------------------------------------------------------
class FactualQA(dspy.Signature):
    """Answer the question with a short, factual answer (a few words at most)."""

    # The docstring above IS the base instruction the optimizer can build on.
    # InputField / OutputField describe the I/O contract; the `desc` text is
    # surfaced to the model, so treat it like lightweight prompt copy.
    question = dspy.InputField(desc="a factual question")
    answer = dspy.OutputField(desc="the concise factual answer, no full sentence")


# ---------------------------------------------------------------------------
# 3. Wrap the signature in a Module (the "program" we optimize)
# ---------------------------------------------------------------------------
class QAProgram(dspy.Module):
    """A one-step Q&A program.

    Using ChainOfThought (instead of plain Predict) means the model reasons
    before answering. The optimizer will later inject worked examples of that
    reasoning as few-shot demonstrations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(FactualQA)

    def forward(self, question: str) -> dspy.Prediction:
        # A Module's forward() is just normal Python; you can chain multiple
        # sub-modules here. We only have one step.
        return self.generate_answer(question=question)


# ---------------------------------------------------------------------------
# 4a. Data: a tiny labeled dataset
# ---------------------------------------------------------------------------
def build_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Return (trainset, devset).

    In real life this comes from a file or DB. Each dspy.Example holds the
    fields named in the signature. `.with_inputs("question")` tells DSPy which
    fields are inputs; everything else (here, `answer`) is treated as a label.
    """
    raw = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many continents are there on Earth?", "7"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("In what year did World War II end?", "1945"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the tallest mountain in the world?", "Mount Everest"),
        ("What gas do plants primarily absorb for photosynthesis?", "Carbon dioxide"),
    ]

    examples = [
        dspy.Example(question=q, answer=a).with_inputs("question")
        for q, a in raw
    ]

    # Split: a handful to bootstrap demos from, the rest to measure on.
    trainset = examples[:6]
    devset = examples[6:]
    return trainset, devset


# ---------------------------------------------------------------------------
# 4b. Metric: how we score a prediction
# ---------------------------------------------------------------------------
def answer_exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Deterministic correctness check.

    A normalized comparison is far more defensible than an LLM-as-judge for
    factual answers: no second model, no drift, fully reproducible. We normalize
    case/whitespace/punctuation and accept the gold answer appearing inside the
    prediction (handles "The capital is Paris." vs "Paris").

    BootstrapFewShot calls this in two ways:
      - during compilation (with a `trace`) to decide whether a candidate
        demonstration is good enough to keep,
      - during evaluation (trace=None) to score dev examples.
    Returning a bool works for both.
    """
    def normalize(text: str) -> str:
        text = text.lower().strip()
        for ch in ".,!?;:'\"":
            text = text.replace(ch, "")
        return " ".join(text.split())

    gold = normalize(example.answer)
    predicted = normalize(pred.answer)
    return gold == predicted or gold in predicted


# ---------------------------------------------------------------------------
# 5. Baseline -> optimize -> compare
# ---------------------------------------------------------------------------
def main() -> None:
    configure_lm()
    trainset, devset = build_dataset()

    # A reusable evaluator over the dev split. This is what gives us the
    # before/after number that makes the optimization visible.
    evaluate = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,
        display_progress=True,
    )

    # --- Baseline: the un-optimized program (zero-shot, just the signature) ---
    program = QAProgram()
    print("\n=== Baseline (no demonstrations) ===")
    baseline_score = evaluate(program)

    # --- Optimize with BootstrapFewShot ---------------------------------------
    # BootstrapFewShot runs the program on the trainset, keeps the traces where
    # the metric passes, and turns those into few-shot demonstrations attached
    # to the module. It's the simplest DSPy optimizer: no instruction rewriting,
    # just automatically curated examples.
    optimizer = BootstrapFewShot(
        metric=answer_exact_match,
        max_bootstrapped_demos=4,   # self-generated demos to include
        max_labeled_demos=4,        # raw trainset demos it may also use
    )

    print("\n=== Compiling with BootstrapFewShot ===")
    optimized_program = optimizer.compile(student=program, trainset=trainset)

    # --- Optimized: same dev split, same metric -------------------------------
    print("\n=== Optimized (with bootstrapped demonstrations) ===")
    optimized_score = evaluate(optimized_program)

    # --- Report ---------------------------------------------------------------
    print("\n----------------------------------------")
    print(f"Baseline  accuracy: {baseline_score}")
    print(f"Optimized accuracy: {optimized_score}")
    print("----------------------------------------")

    # See the actual prompt DSPy sent on the last call, demonstrations and all.
    print("\n=== Last prompt sent to the LM ===")
    dspy.inspect_history(n=1)

    # Persist the optimized program (the learned demos) so you can reload it
    # later with QAProgram().load("optimized_qa.json") instead of recompiling.
    optimized_program.save("optimized_qa.json")
    print("\nSaved optimized program to optimized_qa.json")


if __name__ == "__main__":
    main()
