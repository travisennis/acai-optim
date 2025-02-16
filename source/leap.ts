import { dedent } from "@travisennis/acai-core";
import { type LanguageModel, generateText } from "ai";

// biome-ignore lint/style/useNamingConvention: <explanation>
class LEAP {
  private model: LanguageModel;
  private systemPrompt: string | undefined;
  private lowLevelPrinciples: string[];
  private highLevelPrinciples: string[];
  completionTokens: number;

  constructor(model: LanguageModel, systemPrompt?: string) {
    this.model = model;
    this.systemPrompt = systemPrompt;
    this.lowLevelPrinciples = [];
    this.highLevelPrinciples = [];
    this.completionTokens = 0;
  }

  private extractOutput(text: string): string {
    const match = text.match(/<output>(.*?)(?:<\/output>|$)/s);
    return match ? (match[1]?.trim() ?? "") : "";
  }

  private async extractExamplesFromQuery(
    initialQuery: string,
  ): Promise<[string, string][]> {
    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      temperature: 1.0,
      system: this.systemPrompt,
      prompt: dedent`
                Analyze the following query and determine if it contains few-shot examples.
                If it does, extract the examples and their corresponding answers.
                Format the examples as a JSON array of objects, where each object has "question" and "answer" fields.
                If there are no examples, return an empty array.
                Enclose your response within <output></output> tags.
                Do not put any explanation or any other response other than the JSON array within the <output></output> tags.

                Example output format:
                <output>
                [
                    {"question": "What is 2+2?", "answer": "4"},
                    {"question": "What is the capital of France?", "answer": "Paris"}
                ]
                </output>

                Query: ${initialQuery}
            `,
    });

    this.completionTokens += usage.completionTokens;
    const examplesStr = this.extractOutput(text);
    let examples: [string, string][] = [];

    if (examplesStr) {
      try {
        const examplesList = JSON.parse(examplesStr);
        examples = examplesList.map(
          (example: { question: string; answer: string }) => [
            example.question,
            example.answer,
          ],
        );
      } catch (_error) {
        // Failed to parse examples JSON, using empty list
      }
    }

    return examples;
  }

  private async generateMistakes(
    examples: [string, string][],
  ): Promise<[string, string, string, string][]> {
    const mistakes: [string, string, string, string][] = [];

    for (const [question, correctAnswer] of examples) {
      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 4096,
        temperature: 0.7,
        system: this.systemPrompt,
        prompt: dedent`
                    Instruction: Answer the following question step by step. To induce a mistake, 
                    deliberately introduce an error in your reasoning or calculation.
                    Question: ${question}
                    Provide your step-by-step reasoning, then enclose your final answer within <output></output> tags.
                    Think step by step, but make sure to include a mistake.
                `,
      });

      this.completionTokens += usage.completionTokens;
      const generatedReasoning = text;
      const generatedAnswer = this.extractOutput(generatedReasoning);

      if (generatedAnswer !== correctAnswer) {
        mistakes.push([
          question,
          generatedReasoning,
          generatedAnswer,
          correctAnswer,
        ]);
      }
    }

    return mistakes;
  }

  private async generateLowLevelPrinciples(
    mistakes: [string, string, string, string][],
  ): Promise<string[]> {
    for (const [
      question,
      generatedReasoning,
      generatedAnswer,
      correctAnswer,
    ] of mistakes) {
      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 4096,
        temperature: 1.0,
        system: this.systemPrompt,
        prompt: dedent`
                Question: ${question}
                Generated Reasoning: ${generatedReasoning}
                Generated Answer: ${generatedAnswer}
                Correct Answer: ${correctAnswer}
                Instruction: Conduct a thorough analysis of the generated answer in comparison to the
                correct answer. Also observe how the generated reasoning differs from the correct
                reasoning. Identify any discrepancies, misunderstandings, or errors. Provide clear
                insights, principles, or guidelines that can be derived from this analysis to improve
                future responses. We are not focused on this one data point, but rather on the general
                principle.
                Reasoning: <discuss why the generated answer is wrong>
                Insights: Enclose ONLY the principles or insights within <output></output> tags.
            `,
      });

      this.completionTokens += usage.completionTokens;
      this.lowLevelPrinciples.push(this.extractOutput(text));
    }

    return this.lowLevelPrinciples;
  }

  private async generateHighLevelPrinciples(): Promise<string[]> {
    const principlesText = this.lowLevelPrinciples.join("\n");
    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      temperature: 1.0,
      system: this.systemPrompt,
      prompt: dedent`
            Low-level principles: ${principlesText}
            Create a list of *unique* and insightful principles to improve future responses based
            on the analysis above.
            Focus on capturing the essence of the feedback while eliminating redundancies.
            Ensure that each point is clear, concise, and directly derived from the introspection
            results.
            Create a numbered list of principles. Leave specific details in place.
            Limit to at most 8 principles.
            Enclose your list of principles within <output></output> tags.
        `,
    });

    this.completionTokens += usage.completionTokens;
    this.highLevelPrinciples = this.extractOutput(text).split("\n");
    return this.highLevelPrinciples;
  }

  private async applyPrinciples(query: string): Promise<string> {
    const principlesText = this.highLevelPrinciples.join("\n");
    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      temperature: 1.0,
      system: this.systemPrompt,
      prompt: dedent`
                Please answer the following query. Keep in mind these principles:

                ${principlesText}

                Query: ${query}
            `,
    });

    this.completionTokens += usage.completionTokens;
    return text;
  }

  async solve(initialQuery: string): Promise<{
    result: string;
    intermediateSteps: {
      extractedExamples: [string, string][];
      generatedMistakes: [string, string, string, string][];
      lowLevelPrinciples: string[];
      highLevelPrinciples: string[];
      finalResponse: string;
    };
  }> {
    const examples = await this.extractExamplesFromQuery(initialQuery);
    if (examples.length === 0) {
      const finalResponse = await this.applyPrinciples(initialQuery);
      return {
        result: finalResponse,
        intermediateSteps: {
          extractedExamples: [],
          generatedMistakes: [],
          lowLevelPrinciples: [],
          highLevelPrinciples: this.highLevelPrinciples,
          finalResponse,
        },
      };
    }

    const mistakes = await this.generateMistakes(examples);
    await this.generateLowLevelPrinciples(mistakes);
    await this.generateHighLevelPrinciples();
    const finalResponse = await this.applyPrinciples(initialQuery);
    return {
      result: finalResponse,
      intermediateSteps: {
        extractedExamples: examples,
        generatedMistakes: mistakes,
        lowLevelPrinciples: this.lowLevelPrinciples,
        highLevelPrinciples: this.highLevelPrinciples,
        finalResponse,
      },
    };
  }
}

export async function leap({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> {
  const leapSolver = new LEAP(model, system);
  const { result, intermediateSteps } = await leapSolver.solve(prompt);

  const finalResult = dedent`
${JSON.stringify(intermediateSteps, null, 2)}

Result:
${result}`;

  return [finalResult, leapSolver.completionTokens];
}
