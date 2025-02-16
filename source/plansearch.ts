import { type LanguageModel, generateText } from "ai";

interface PlanSearchResult {
  initialObservations: string[];
  derivedObservations: string[];
  naturalLanguageSolution: string;
  implementation: string;
}

function formatPlanSearchResult(result: PlanSearchResult): string {
  // Helper function to format a list of observations with a title
  // Returns empty string if no observations, otherwise formats with bullet points
  const formatObservations = (
    observations: string[],
    title: string,
  ): string => {
    if (!observations.length) return "";
    return `${title}:\n${observations.map((obs) => `  • ${obs}`).join("\n")}\n`;
  };

  // Creates array with following elements:
  // 1. Formatted initial observations
  // 2. Formatted derived observations
  // 3. "Solution:" header
  // 4. The natural language solution
  // 5. "Implementation:" header
  // 6. The implementation details
  return (
    [
      formatObservations(result.initialObservations, "Initial Observations"),
      formatObservations(result.derivedObservations, "Derived Observations"),
      "\nSolution:\n",
      result.naturalLanguageSolution,
      "\nImplementation:\n",
      result.implementation,
    ]
      // filter(Boolean) removes any falsy values (empty strings, null, undefined)
      // This ensures we don't get extra newlines from empty sections
      .filter(Boolean)
      // Join all remaining elements with newlines between them
      .join("\n")
  );
}

interface TokenCount {
  tokens: number;
}

class PlanSearch {
  private model: LanguageModel;
  private systemPrompt: string | undefined;

  constructor(model: LanguageModel, systemPrompt?: string) {
    this.systemPrompt = systemPrompt;
    this.model = model;
  }

  async generateObservations(
    problem: string,
    numObservations = 3,
  ): Promise<{ observations: string[]; tokens: number }> {
    const prompt = `Generate ${numObservations} specific, non-obvious, and correct observations about the following problem. Each observation should be numbered and start on a new line.

Problem:
${problem}`;

    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      system: this.systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });

    // Improved parsing
    const observations = text
      .trim()
      .split("\n")
      .filter((line) => /^\d+\./.test(line.trim())) // Only keep numbered lines
      .map((obs) => obs.replace(/^\d+\.\s*/, "").trim()); // Remove numbers

    return {
      observations: observations.filter((obs) => obs.trim()),
      tokens: usage.completionTokens,
    };
  }

  async generateDerivedObservations(
    problem: string,
    observations: string[],
    numNewObservations = 2,
  ): Promise<{ observations: string[]; tokens: number }> {
    const prompt = `Based on the existing observations, generate ${numNewObservations} new, derived observations about the following problem. Each new observation should build upon or combine insights from the existing observations. Number each new observation.

Problem:
${problem}

Existing observations:
${observations.map((obs, i) => `${i + 1}. ${obs}`).join("\n")}`;

    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      system: this.systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });

    const newObservations = text.trim().split("\n");
    return {
      observations: newObservations.filter((obs) => obs.trim()),
      tokens: usage.completionTokens,
    };
  }

  async generateSolution(
    problem: string,
    observations: string[],
  ): Promise<{ solution: string; tokens: number }> {
    const prompt = `Problem:
${problem}

Relevant observations:
${observations.map((obs, i) => `${i + 1}. ${obs}`).join("\n")}

Using the above observations, construct a step-by-step solution to the problem. For each step:
1. Quote the specific observation(s) that inform that step
2. Explain how the observation leads to the solution step
3. Show your work/reasoning

Conclude with a clear, final answer.`;

    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      system: this.systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });

    return {
      solution: text.trim(),
      tokens: usage.completionTokens,
    };
  }

  async implementSolution(
    problem: string,
    solution: string,
  ): Promise<{ implementation: string; tokens: number }> {
    const prompt = `Problem:
${problem}

Solution specification:
${solution}

Generate a concrete implementation of the solution above. The implementation should:
1. Be directly executable/applicable
2. Follow all specifications from the solution
3. Be as concise as possible while maintaining clarity`;

    const { text, usage } = await generateText({
      model: this.model,
      maxTokens: 4096,
      system: this.systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });

    return {
      implementation: text.trim(),
      tokens: usage.completionTokens,
    };
  }

  async solve(
    problem: string,
    numInitialObservations = 3,
    numDerivedObservations = 2,
  ): Promise<PlanSearchResult & TokenCount> {
    let totalTokens = 0;

    const { observations: initial, tokens: t1 } =
      await this.generateObservations(problem, numInitialObservations);
    totalTokens += t1;

    const { observations: derived, tokens: t2 } =
      await this.generateDerivedObservations(
        problem,
        initial,
        numDerivedObservations,
      );
    totalTokens += t2;

    const allObservations = [...initial, ...derived];
    const { solution, tokens: t3 } = await this.generateSolution(
      problem,
      allObservations,
    );
    totalTokens += t3;

    const { implementation, tokens: t4 } = await this.implementSolution(
      problem,
      solution,
    );
    totalTokens += t4;

    return {
      initialObservations: initial,
      derivedObservations: derived,
      naturalLanguageSolution: solution,
      implementation,
      tokens: totalTokens,
    };
  }

  async solveMultiple(
    problem: string,
    n: number,
    numInitialObservations = 3,
    numDerivedObservations = 2,
  ): Promise<{
    attempts: Array<PlanSearchResult>;
    tokens: number;
  }> {
    const attempts: Array<PlanSearchResult> = [];
    let totalTokens = 0;

    for (let i = 0; i < n; i++) {
      const result = await this.solve(
        problem,
        numInitialObservations,
        numDerivedObservations,
      );
      totalTokens += result.tokens;
      attempts.push({
        initialObservations: result.initialObservations,
        derivedObservations: result.derivedObservations,
        naturalLanguageSolution: result.naturalLanguageSolution,
        implementation: result.implementation,
      });
    }

    return {
      attempts,
      tokens: totalTokens,
    };
  }
}

export async function plansearch({
  model,
  system,
  prompt,
  n = 1,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
  n?: number;
}): Promise<[string, number]> {
  const planner = new PlanSearch(model, system);
  const result = await planner.solveMultiple(prompt, n);
  const response = result.attempts
    .map((attempt) => formatPlanSearchResult(attempt))
    .join("\n\n");
  return [response, result.tokens];
}
