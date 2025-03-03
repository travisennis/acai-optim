import { type LanguageModel, generateText } from "ai";

class ChainOfDraft {
  private model: LanguageModel;
  private systemPrompt?: string;

  constructor(model: LanguageModel, systemPrompt?: string) {
    this.model = model;
    this.systemPrompt = systemPrompt;
  }

  async send(initialQuery: string) {
    const codPrompt = `Think step by step, keeping each step as a minimum draft (max 5 words). Then, answer the question:\n\n${initialQuery}`;

    // Make the API call
    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 8096,
      system: this.systemPrompt,
      prompt: codPrompt,
    });

    return { text: text, completionTokens: usage.completionTokens };
  }
}

export async function cod({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> {
  const instance = new ChainOfDraft(model, system);
  const result = await instance.send(prompt);

  return [result.text, result.completionTokens];
}
