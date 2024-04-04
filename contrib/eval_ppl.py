import random

import glog
import torch
from tqdm import tqdm

from . import gptq_data_utils
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)


def main(datasets, model_str, quantized_path, seed, seqlen):
    print(quantized_path)
    model = AutoModelForCausalLM.from_pretrained(quantized_path, device_map="auto")

    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=seed,
                                                    seqlen=seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(
            nsamples, seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                           use_cache=False,
                           output_hidden_states=False,
                           output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        glog.info(f"{dataset} perplexity: {ppl}")


if __name__ == "__main__":
    datasets = ["wikitext2", "c4"]
    model_str = "meta-llama/Llama-2-7b-hf"
    quantized_path = "/share/desa/nfs01/qs234/checkpoints/meta-llama/Llama-2-7b-hf-none"
    seed = 42
    seqlen = 4096

    torch.set_grad_enabled(False)
    random.seed(seed)
    torch.random.manual_seed(seed)
    main(datasets, model_str, quantized_path, seed, seqlen)
