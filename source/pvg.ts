import { dedent } from "@travisennis/acai-core";
import { type LanguageModel, generateText } from "ai";

// biome-ignore lint/style/useNamingConvention: <explanation>
export class PVG {
  private model: LanguageModel;
  private systemPrompt: string;
  completionTokens = 0;

  constructor(model: LanguageModel, systemPrompt?: string) {
    this.model = model;
    this.systemPrompt = systemPrompt ?? "";
  }

  async generateSolutions(
    query: string,
    numSolutions: number,
    isSneaky = false,
    temperature = 0.7,
  ): Promise<string[]> {
    const role = isSneaky ? "sneaky" : "helpful";

    const roleInstruction = dedent`
      You are a problem solver tasked with generating solutions to the given problem. 
      If you are in 'helpful' mode:
      - Provide a correct, detailed, and step-by-step solution to the problem.
      - Explain your reasoning clearly and logically.
      - Ensure your final answer or conclusion is accurate and well-supported.
      - Use domain-specific terminology or notation if appropriate.

      If you are in 'sneaky' mode:
      - Provide a solution that appears correct but contains subtle errors or flaws.
      - Make your explanation seem logical and convincing.
      - Ensure your final answer or conclusion is incorrect, but in a way that's not immediately obvious.
      - Maintain the appearance of expertise in the relevant domain.
      `;

    const solutions: string[] = [];
    for (let i = 0; i < numSolutions; i++) {
      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 4096,
        temperature,
        system: `${this.systemPrompt}\n${roleInstruction}\nYou are in ${role} mode.`,
        prompt: query,
      });
      this.completionTokens += usage.completionTokens;
      solutions.push(text);
    }

    return solutions;
  }

  async verifySolutions(
    initialQuery: string,
    solutions: string[],
  ): Promise<{
    scores: number[];
    verificationDetails: {
      explanations: string[];
      criteria: {
        accuracy: number;
        clarity: number;
        completeness: number;
      }[];
    };
  }> {
    const verifyPrompt = dedent`${this.systemPrompt}
You are a verifier tasked with evaluating the correctness and clarity of solutions to the given problem.
Rate the following solution on a scale from 0 to 10, where:
- 0 is completely incorrect or incomprehensible
- 5 is partially correct or lacks clarity
- 10 is perfectly correct, clear, and well-explained

Consider the following criteria:
1. Accuracy of the final answer or conclusion
2. Correctness of each step or argument in the solution
3. Clarity and logical flow of the explanation
4. Appropriate use of domain-specific concepts or terminology
5. Completeness of the solution

Be especially vigilant for subtle errors or flaws that might not be immediately obvious.

Provide your response in the following format:

Score: [Your numerical score from 0 to 10]
Explanation: [Your detailed explanation for the score, highlighting specific strengths or weaknesses]

Ensure that the Score is a single number between 0 and 10, and the Explanation is on a new line.`;

    const scores: number[] = [];
    const explanations: string[] = [];
    for (const solution of solutions) {
      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 1024,
        temperature: 0.2,
        system: verifyPrompt,
        prompt: `Problem: ${initialQuery}\n\nSolution: ${solution}`,
      });

      this.completionTokens += usage.completionTokens;

      const scoreMatch = text.match(/Score:\s*(\d+(\.\d+)?)/);
      const explanationMatch = text.match(/Explanation:\s*(.*)/s);

      if (scoreMatch) {
        try {
          const score = Number.parseFloat(scoreMatch[1] ?? "");
          scores.push(score);
          if (explanationMatch) {
            const explanation = explanationMatch[1]?.trim() ?? "";
            explanations.push(explanation);
          }
        } catch (_error) {
          scores.push(0);
        }
      } else {
        scores.push(0);
      }
    }

    return {
      scores,
      verificationDetails: {
        explanations,
        criteria: solutions.map(() => ({
          accuracy: Math.random() * 10, // These would ideally be parsed from the explanation
          clarity: Math.random() * 10,
          completeness: Math.random() * 10,
        })),
      },
    };
  }

  get tokens(): number {
    return this.completionTokens;
  }
}

export async function pvg({
  model,
  systemPrompt,
  prompt,
  numRounds = 2,
  numSolutions = 3,
}: {
  model: LanguageModel;
  systemPrompt?: string;
  prompt: string;
  numRounds?: number;
  numSolutions?: number;
}): Promise<[string, number]> {
  const pvg = new PVG(model, systemPrompt);
  let bestSolution = "";
  let bestScore = -1;
  let currentPrompt = prompt;
  let response = "";
  for (let round = 0; round < numRounds; round++) {
    response += `Round ${round + 1}\n\n`;

    const temperature = Math.max(0.2, 0.7 - round * 0.1);

    const helpfulSolutions = await pvg.generateSolutions(
      currentPrompt,
      numSolutions,
      false,
      temperature,
    );
    const sneakySolutions = await pvg.generateSolutions(
      currentPrompt,
      numSolutions,
      true,
      temperature,
    );
    const allSolutions = [...helpfulSolutions, ...sneakySolutions];

    response += `Generation Details:\n${JSON.stringify(
      {
        helpful: helpfulSolutions,
        sneaky: sneakySolutions,
      },
      null,
      2,
    )}\n\n`;

    const { scores, verificationDetails } = await pvg.verifySolutions(
      currentPrompt,
      allSolutions,
    );

    response += `Scores:\n${JSON.stringify(scores, null, 2)}\n\n`;
    response += `Verification Details:\n${JSON.stringify(verificationDetails, null, 2)}\n\n`;

    const roundBestIndex = scores.indexOf(Math.max(...scores));
    const roundBestScore = scores[roundBestIndex] ?? 0;
    const roundBestSolution = allSolutions[roundBestIndex];

    if (roundBestScore > bestScore) {
      bestSolution = roundBestSolution ?? "";
      bestScore = roundBestScore;
    }

    response += `Current best solution:\n${bestSolution}`;

    if (round < numRounds - 1) {
      const refinePrompt = dedent`
        Based on the original query and the best solution so far, suggest a refined query that might lead to an even better solution.
        Focus on aspects of the problem that were challenging or not fully addressed in the best solution.
        Maintain the core intent of the original query while adding specificity or context that could improve the solution.
        
        Original query: ${currentPrompt}
        
        Best solution so far: ${bestSolution}
        
        Refined query:
        `;

      const { text, usage } = await generateText({
        model,
        maxTokens: 1024,
        temperature: 0.5,
        system: systemPrompt,
        prompt: refinePrompt,
      });

      pvg.completionTokens += usage.completionTokens;
      currentPrompt = text;

      response += `Refined prompt:\n${currentPrompt}\n\n`;
    }
  }

  response += `Final solution:\n${bestSolution}`;

  return [response, pvg.tokens];
}
