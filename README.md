# EchoLearn – Prototype Notebook

This notebook contains an experimental prototype for a local AI-based study assistant using the Gemma 3n model via Ollama.

## Purpose

The goal was to test whether a local LLM could:
- Summarize educational text
- Generate relevant quiz questions
- Grade and give feedback on student answers

## Structure

- **Text Preprocessing**: Splits raw text into sections using a heuristic-based method
- **Prompt Chaining**: Sends each section to Gemma with instructions to summarize and generate quiz questions
- **Answer Grading**: Sends user answers to Gemma with grading prompts to simulate feedback

## Dependencies

- Python 3.8+
- [Ollama](https://ollama.com) with `gemma:3n` model installed

## Notes

- All code is contained in a Jupyter notebook for simplicity and testing.
- Uses structured prompts to interact with Gemma 3n locally.
- Output is parsed from JSON format and used to simulate summarization, quiz generation, and grading workflows.

## Status

This notebook serves as a proof-of-concept for local summarization, quiz generation, and grading using a lightweight LLM. Additional features, such as audio input or a user interface, could be explored in future iterations.
