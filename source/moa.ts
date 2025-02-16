import { dedent } from "@travisennis/acai-core";
import { type LanguageModel, generateText } from "ai";

export async function moa({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> {
  let moaCompletionTokens = 0;

  const initialResponse = await Promise.all(
    new Array(3).fill(0).map((_) => {
      return generateText({
        model: model,
        maxTokens: 4096,
        temperature: 1.0,
        system: system,
        prompt: prompt,
      });
    }),
  );

  const completions = initialResponse.map((resp) => resp.text);
  moaCompletionTokens += initialResponse.reduce((prev, curr) => {
    return prev + curr.usage.completionTokens;
  }, 0);

  const critiquePrompt = dedent`
    Original query: ${prompt}

    I will present you with three candidate responses to the original query. Please analyze and critique each response, discussing their strengths and weaknesses. Provide your analysis for each candidate separately.

    Candidate 1:
    ${completions[0]}

    Candidate 2:
    ${completions[1]}

    Candidate 3:
    ${completions[2]}

    Please provide your critique for each candidate:
    `;

  const critiqueResponse = await generateText({
    model: model,
    maxTokens: 512,
    temperature: 0.1,
    system: system,
    prompt: critiquePrompt,
  });

  const critiques = critiqueResponse.text;
  moaCompletionTokens += critiqueResponse.usage.completionTokens;

  const finalPrompt = dedent`
    Original query: ${prompt}

    Based on the following candidate responses and their critiques, generate a final response to the original query.

    Candidate 1:
    ${completions[0]}

    Candidate 2:
    ${completions[1]}

    Candidate 3:
    ${completions[2]}

    Critiques of all candidates:
    ${critiques}

    Please provide a final, optimized response to the original query:
    `;

  const finalResponse = await generateText({
    model: model,
    maxTokens: 8192,
    temperature: 0.1,
    system: system,
    prompt: finalPrompt,
  });

  const result = dedent`
    Candidate 1:
    ${completions[0]}

    Candidate 2:
    ${completions[1]}

    Candidate 3:
    ${completions[2]}

    Critiques of all candidates:
    ${critiques}

    Final response:
    ${finalResponse.text}
`;

  moaCompletionTokens += finalResponse.usage.completionTokens;

  return [result, moaCompletionTokens];
}
