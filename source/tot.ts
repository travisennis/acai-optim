import { type LanguageModel, generateObject, generateText } from "ai";
import { z } from "zod";

class ThoughtNode {
  children: ThoughtNode[] = [];
  content: string;
  score: number;

  constructor(content: string, score = 0) {
    this.content = content;
    this.score = score;
  }

  addChild(child: ThoughtNode): void {
    this.children.push(child);
  }
}

class TreeOfThought {
  private model: LanguageModel;
  private system: string | undefined;
  private root: ThoughtNode;
  private maxDepth: number;
  private branchingFactor: number;
  // Track all explored thoughts and their scores
  private exploredThoughts: Array<{
    thought: string;
    score: number;
    depth: number;
  }> = [];
  usageTokens: number;

  constructor({
    model,
    system,
    prompt,
    maxDepth = 3,
    branchingFactor = 3,
  }: {
    model: LanguageModel;
    system?: string;
    prompt: string;
    maxDepth?: number;
    branchingFactor?: number;
  }) {
    this.model = model;
    this.system = system;
    this.root = new ThoughtNode(prompt);
    this.maxDepth = maxDepth;
    this.branchingFactor = branchingFactor;
    this.usageTokens = 0;
  }

  private async generateThought(parentThought: string): Promise<string> {
    const { text, usage } = await generateText({
      model: this.model,
      temperature: 0.7,
      maxTokens: 4096,
      system: this.system,
      prompt: `Next thought based on: ${parentThought}`,
    });
    this.usageTokens += usage.completionTokens;
    return text;
  }

  private async evaluateThought(thought: string): Promise<number> {
    const { object, usage } = await generateObject({
      model: this.model,
      temperature: 0.7,
      maxTokens: 200,
      schema: z.object({ score: z.number() }),
      prompt: `Evaluate: ${thought}`,
    });

    this.exploredThoughts.push({
      thought,
      score: object.score,
      depth: this.exploredThoughts.length,
    });

    this.usageTokens += usage.completionTokens;
    return object.score;
  }

  private async expand(node: ThoughtNode, depth: number): Promise<void> {
    if (depth >= this.maxDepth) {
      return;
    }

    for (let i = 0; i < this.branchingFactor; i++) {
      const newThought = await this.generateThought(node.content);
      const childNode = new ThoughtNode(newThought);
      node.addChild(childNode);
      await this.expand(childNode, depth + 1);
    }
  }

  async findBestSolution(): Promise<[string, string]> {
    await this.expand(this.root, 0);
    const queue: ThoughtNode[] = [this.root];
    let bestNode = this.root;

    while (queue.length > 0) {
      const node = queue.shift();
      if (!node) {
        continue;
      }

      node.score = await this.evaluateThought(node.content);
      if (node.score > bestNode.score) {
        bestNode = node;
      }
      queue.push(...node.children);
    }

    // Format the result to show the exploration process
    const result = `Exploration Process:\n${this.exploredThoughts
      .map(
        (t) => `Depth ${t.depth}:\nThought: ${t.thought}\nScore: ${t.score}\n`,
      )
      .join("\n")}\n\nBest Solution:\n${bestNode.content}`;

    return [result, bestNode.content];
  }
}

export async function tot({
  model,
  system,
  prompt,
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
}): Promise<[string, number]> {
  const instance = new TreeOfThought({ model, system, prompt });

  const [result, _bestSolution] = await instance.findBestSolution();

  return [result, instance.usageTokens];
}
