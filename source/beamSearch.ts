import { generateText, type LanguageModel } from "ai";

interface BeamNode {
  text: string;
  score: number;
  finished: boolean;
}

class BeamSearch {
  private model: LanguageModel;
  private beamWidth: number;
  private maxLength: number;
  private temperature: number;
  completionTokens: number;

  constructor(
    model: LanguageModel,
    beamWidth: number,
    maxLength: number,
    temperature: number,
  ) {
    this.model = model;
    this.beamWidth = beamWidth;
    this.maxLength = maxLength;
    this.temperature = temperature;
    this.completionTokens = 0;
  }

  async generate(prompt: string): Promise<string[]> {
    // Initialize beam with the first node
    let beams: BeamNode[] = [
      {
        text: prompt,
        score: 0,
        finished: false,
      },
    ];

    // Keep track of completed sequences
    const completedSequences: BeamNode[] = [];

    while (beams.length > 0 && completedSequences.length < this.beamWidth) {
      const candidates: BeamNode[] = [];

      // Expand each beam
      for (const beam of beams) {
        if (beam.finished) {
          completedSequences.push(beam);
          continue;
        }

        try {
          // Get next token predictions using Anthropic API
          const continuations = await Promise.all(
            new Array(this.beamWidth).fill(0).map((_) => {
              return generateText({
                model: this.model,
                maxTokens: 5,
                temperature: this.temperature,
                prompt: `${beam.text}\n\nContinue this text with a few words. It does not have to be a complete thought or sentence. Just the most likely continuation of the provided text. Do not include the provided text.`,
              });
            }),
          );

          // Create new beam nodes for each continuation
          for (const continuation of continuations) {
            this.completionTokens += continuation.usage.completionTokens;
            const continuationText = continuation.text;

            const newText = `${beam.text} ${continuationText}`;

            // Calculate a simple score (you might want to implement a more sophisticated scoring mechanism)
            const newScore = beam.score - Math.log(continuationText.length);

            // Check if sequence should be terminated
            const shouldTerminate =
              newText.length >= this.maxLength ||
              newText.includes(".") ||
              newText.includes("!") ||
              newText.includes("?");

            candidates.push({
              text: newText,
              score: newScore,
              finished: shouldTerminate,
            });
          }
        } catch (error) {
          console.error("Error during beam search:", error);
        }
      }

      // Sort candidates by score and keep top-k beams
      beams = candidates
        .sort((a, b) => b.score - a.score)
        .slice(0, this.beamWidth)
        .filter((beam) => !beam.finished);

      // Add finished sequences to completed sequences
      completedSequences.push(...candidates.filter((beam) => beam.finished));
    }

    // Return the top completed sequences
    return completedSequences
      .sort((a, b) => b.score - a.score)
      .slice(0, this.beamWidth)
      .map((beam) => beam.text);
  }
}
export async function beamSearch({
  model,
  prompt,
  beamWidth = 3,
  maxLength = 150,
  temperature = 0.7,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
  beamWidth?: number;
  maxLength?: number;
  temperature?: number;
}): Promise<[string, number]> {
  const beamSearch = new BeamSearch(model, beamWidth, maxLength, temperature);

  const results = await beamSearch.generate(prompt);

  let result = "Generated sequences:\n";
  results.forEach((sequence, index) => {
    result += `${index + 1}. ${sequence}\n`;
  });

  return [result, beamSearch.completionTokens];
}
