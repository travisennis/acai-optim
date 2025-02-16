import { type LanguageModel, generateText } from "ai";

export async function bon({
  model,
  system,
  prompt,
  n = 3,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
  n?: number;
}): Promise<[string, number]> {
  let tokens = 0;

  const completions = await Promise.all(
    new Array(n).fill(0).map(async (_) => {
      const { text, usage } = await generateText({
        model: model,
        maxTokens: 4096,
        temperature: 1.0,
        system,
        prompt,
      });
      tokens += usage.completionTokens;
      return text;
    }),
  );

  const ratings = await Promise.all(
    completions.map(async (completion) => {
      const { text, usage } = await generateText({
        model: model,
        maxTokens: 256,
        temperature: 0.1,
        system:
          "You are to be given a query and response to that query. Rate the response on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness to the query. Respond only with a number.",
        prompt: `Query: ${prompt}\n\nResponse: ${completion}`,
      });
      tokens += usage.completionTokens;
      return Number.parseInt(text);
    }),
  );

  const bestIndex = ratings.indexOf(Math.max(...ratings));

  const result = `${zip(completions, ratings)
    .map(
      (item, idx) => `Candidate ${idx + 1}:\n${item[0]}\n\nRating:${item[1]}`,
    )
    .join("\n\n")}n\nBest Result:\n${completions[bestIndex]}`;

  return [result, tokens];
}

function zip<T, U>(arr1: T[], arr2: U[]): [T, U][] {
  const length = Math.min(arr1.length, arr2.length);
  const result: [T, U][] = [];

  for (let i = 0; i < length; i++) {
    const arr1Value = arr1[i];
    const arr2Value = arr2[i];
    if (arr1Value && arr2Value) {
      result.push([arr1Value, arr2Value]);
    }
  }

  return result;
}
