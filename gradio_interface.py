import fire
import gradio as gr
from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def chat_with_llama(generator, user_input):
    dialogs = [
        [{"role": "user", "content": user_input}],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return results[0]['generation']['content']

if __name__ == "__main__":
    generator = fire.Fire(main)
    iface = gr.Interface(fn=chat_with_llama, inputs="text", outputs="text")
    iface.launch()
