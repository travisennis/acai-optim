import { type LanguageModel, generateText } from "ai";

export const reread = async ({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> => {
  // Construct the RE2 prompt
  const re2Prompt = `${prompt}\nRead the question again: ${prompt}`;

  try {
    const { text, usage } = await generateText({
      model,
      maxTokens: 4096,
      temperature: 0.1,
      system,
      prompt: re2Prompt,
    });

    const re2CompletionTokens = usage.completionTokens;

    return [text.trim(), re2CompletionTokens];
  } catch (error) {
    console.error(`Error in RE2 approach: ${error}`);
    throw error;
  }
};
