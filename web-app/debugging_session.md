changing ['input'] to ['input_values'] in session.handler.inputNames fixes the problem

Q: what sets up session?

the next immediate problem is that CausalLMOutput constructor cares only about logits, which is not present, but I need to figure out a way to extract what else is being passed there, because maybe it needs a fix

in my case, the output data seem to be in output.cpuData or smth like that

then

```
    for (const aud of preparedAudios) {
      const inputs = await this.processor(aud);
      const output = await this.model(inputs);
      const logits = output.logits[0];
      const predicted_ids = [];
      for (const item of logits) {
        predicted_ids.push(max(item.data)[1]);
      }
      const predicted_sentences = this.tokenizer.decode(predicted_ids);
      toReturn.push({ text: predicted_sentences });
    }
    return single ? toReturn[0] : toReturn;
  }
```

in this code, if output contained this cpuData or something, then I could do decoding
