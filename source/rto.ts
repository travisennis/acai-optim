import { type LanguageModel, generateText } from "ai";

export const roundTripOptimization = async ({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> => {
  let rtoCompletionTokens = 0;

  // Generate initial code (C1)
  const { text: c1Response, usage: c1Usage } = await generateText({
    model,
    maxTokens: 4096,
    temperature: 0.1,
    system: system,
    prompt: prompt,
  });
  rtoCompletionTokens += c1Usage.completionTokens;

  // Generate description of the response (Q2)
  const { text: q2Response, usage: q2Usage } = await generateText({
    model,
    maxTokens: 1024,
    temperature: 0.1,
    system: system,
    prompt: `Read and understand the following text. Generate an instruction such that if you were given the instruction you could recreate the text yourself.

<text>
${c1Response}
</text>
      `,
  });
  rtoCompletionTokens += q2Usage.completionTokens;

  // Generate second response based on the description (C2)
  const { text: c2Response, usage: c2Usage } = await generateText({
    model,
    maxTokens: 4096,
    temperature: 0.1,
    system: system,
    prompt: q2Response,
  });
  rtoCompletionTokens += c2Usage.completionTokens;

  // Generate final version (C3)
  const finalPrompt = `Initial query: ${prompt}\n\nFirst generated response (C1):\n${c1Response}\n\nSecond generated response (C2):\n${c2Response}\n\nBased on the initial query and these two different responses, generate a final response.`;
  const { text: c3Response, usage: c3Usage } = await generateText({
    model,
    maxTokens: 4096,
    temperature: 0.1,
    system: prompt,
    prompt: finalPrompt,
  });

  const result = `First generated response (C1):\n${c1Response}\n\nDerived query:\n${q2Response}\n\nSecond generated response (C2):\n${c2Response}\n\nFinal response:\n${c3Response}`;

  rtoCompletionTokens += c3Usage.completionTokens;

  return [result, rtoCompletionTokens];
};
