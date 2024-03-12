# llm-bedrock-mistral
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)]()

Plugin for [LLM](https://llm.datasette.io/) adding support for Anthropic's Claude models.

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
# llm install llm-bedrock-mistral -> i just code for some minutes and not much time to do this :()
pip install ./
```
## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-west-2` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_PROFILE=personal
```

## Usage

This plugin adds models called `bedrock-claude` and `bedrock-claude-instant`.

You can query them like this:

```bash
llm -m bedrock-claude-instant "Ten great names for a new space station"
```

```bash
llm -m bedrock-claude "Compare and contrast the leadership styles of Abraham Lincoln and Boris Johnson."
```

## Options

- `max_tokens`, default 1024: The maximum number of tokens to generate before stopping
- `top_p`, default 1: Nucleus sampling, where the model considers the tokens with top_p probability mass
- `temperature`, default 0.7: Determines the sampling temperature. Higher values like 0.8 increase randomness,

Use like this:
```bash
llm -m bedrock-mistral-7b -o max_tokens 200 "Sing me the alphabet"
 I'd be happy to help you recite the alphabet! Here it is:

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.

Is there a particular song or melody you'd like me to sing the alphabet to? I can try my best!
```
