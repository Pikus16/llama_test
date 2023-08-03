"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import pandas as pd

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    csv_file_path: str,  # Added csv_file_path as an argument
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Read prompts from CSV file using pandas
    df = pd.read_csv(csv_file_path)
    prompts = df['prompts'].tolist()

    # Pass the list of prompts to the 'text_completion' method of the 'Llama' class
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Create a new DataFrame to store the prompts and their corresponding completions
    df_results = pd.DataFrame(list(zip(prompts, results)), columns=['prompts', 'completions'])

    # Write the DataFrame to a new CSV file
    df_results.to_csv('completions.csv', index=False)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
"""
