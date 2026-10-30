# Manimatic (original prototype)

A prototype that turns a maths question into an explanation, a [Manim](https://www.manim.community/)
animation and a spoken audio track. It was written as a university project; the write-up is in
[`Manimatic - A math visualiser and explainer.pdf`](<./Manimatic - A math visualiser and explainer.pdf>).

The later project with the same goal, a Next.js and FastAPI video-generation pipeline, is `Manimatic-3.0-V2`.

## How `Model.py` works

It runs a local Llama-2 model (GGUF, through `llama.cpp` and LangChain) in three chained steps:

1. **Explain**: a detailed explanation of the concept you type in.
2. **Visualise**: a description of how to visualise each part of that explanation.
3. **Code**: Manim Community v0.14.0 code that animates it.

It then runs the generated Manim code and saves the explanation as an MP3 with gTTS.

## Before you run it

- **It executes model-generated code.** Step 3 passes the LLM's output straight to `exec`.
  Only run it in a throwaway environment, and read the generated code if you can.
- `MODEL_PATH` points at a Llama-2 `.gguf` file on the original author's machine. Download a
  model and edit the path.
- It needs a machine able to run the model, plus `torch`, `gtts`, `langchain`, `llama-cpp-python` and `manim`.
- The script was written against the LangChain 0.0.x import paths, so newer versions need the imports updated.
