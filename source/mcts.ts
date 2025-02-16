import { randomUUID } from "node:crypto";
import { type CoreMessage, type LanguageModel, generateText } from "ai";

interface EvaluationMetrics {
  coherence: number;
  relevance: number;
  engagement: number;
}

// Immutable state representation for better tree search
class DialogueState {
  readonly conversationHistory: ReadonlyArray<CoreMessage>;
  readonly systemPrompt: string;
  readonly currentQuery: string;
  readonly depth: number;
  metrics?: EvaluationMetrics;

  private constructor(
    systemPrompt: string,
    conversationHistory: ReadonlyArray<CoreMessage>,
    currentQuery: string,
    depth = 0,
    metrics?: EvaluationMetrics,
  ) {
    this.systemPrompt = systemPrompt;
    this.conversationHistory = conversationHistory;
    this.currentQuery = currentQuery;
    this.depth = depth;
    this.metrics = metrics;
  }

  static create(
    systemPrompt: string,
    history: CoreMessage[],
    query: string,
    depth = 0,
    metrics?: EvaluationMetrics,
  ): DialogueState {
    return new DialogueState(
      systemPrompt,
      Object.freeze([...history]),
      query,
      depth,
      metrics,
    );
  }

  withNewMessage(role: "user" | "assistant", content: string): DialogueState {
    return DialogueState.create(
      this.systemPrompt,
      [...this.conversationHistory, { role, content }],
      this.currentQuery,
      this.depth + 1,
      this.metrics,
    );
  }

  getLastResponse(): string | null {
    const lastMessage =
      this.conversationHistory[this.conversationHistory.length - 1];
    return lastMessage?.role === "assistant"
      ? (lastMessage.content as string)
      : null;
  }

  // Hash function for state comparison
  hash(): string {
    return JSON.stringify({
      history: this.conversationHistory,
      query: this.currentQuery,
      depth: this.depth,
    });
  }
}

class MCTSNode {
  readonly id: string;
  readonly state: DialogueState;
  readonly parent: MCTSNode | null;
  readonly action: string | null;
  readonly children: MCTSNode[];
  visits: number;
  isFullyExpanded: boolean;
  private totalValue: number;
  private raveVisits: number;
  private raveValue: number;

  constructor(
    state: DialogueState,
    parent: MCTSNode | null = null,
    action: string | null = null,
    visits = 0,
    totalValue = 0,
    isFullyExpanded = false,
  ) {
    this.id = randomUUID();
    this.state = state;
    this.parent = parent;
    this.action = action;
    this.children = [];
    this.visits = visits;
    this.isFullyExpanded = isFullyExpanded;
    this.totalValue = totalValue;
    this.raveVisits = 0;
    this.raveValue = 0;
  }

  addVisit(value: number): void {
    this.visits++;
    this.totalValue += value;
  }

  addRaveData(value: number): void {
    this.raveVisits++;
    this.raveValue += value;
  }

  getVisits(): number {
    return this.visits;
  }

  getTotalValue(): number {
    return this.totalValue;
  }

  getAverageValue(): number {
    return this.visits === 0 ? 0 : this.totalValue / this.visits;
  }

  getRaveValue(): number {
    return this.raveVisits === 0 ? 0 : this.raveValue / this.raveVisits;
  }

  // PUCT (Predictor + UCT) formula for selection
  getPUCTScore(totalParentVisits: number, priorProbability: number): number {
    const C = 1.5; // Exploration constant
    const exploitation = this.visits === 0 ? 0 : this.totalValue / this.visits;
    const exploration =
      (C * priorProbability * Math.sqrt(totalParentVisits)) / (1 + this.visits);

    // Mix RAVE with UCT
    const beta =
      this.visits /
      (this.visits +
        this.raveVisits +
        4 * priorProbability * totalParentVisits);
    const raveComponent = this.getRaveValue();

    return beta * exploitation + (1 - beta) * raveComponent + exploration;
  }
}

class MCTS {
  private readonly transpositionTable: Map<string, MCTSNode>;
  private readonly actionPriors: Map<string, number>;
  private readonly model: LanguageModel;
  private readonly simulationDepth: number;
  private readonly numSimulations: number;
  private readonly options: {
    maxDepth: number;
    maxChildren: number;
    explorationConstant: number;
    temperature: number;
    useRAVE: boolean;
  };
  completionTokens: number;

  constructor(
    model: LanguageModel,
    simulationDepth: number,
    numSimulations: number,
    options: {
      maxDepth: number;
      maxChildren: number;
      explorationConstant: number;
      temperature: number;
      useRAVE: boolean;
    },
  ) {
    this.model = model;
    this.simulationDepth = simulationDepth;
    this.numSimulations = numSimulations;
    this.options = options;
    this.completionTokens = 0;
    this.transpositionTable = new Map();
    this.actionPriors = new Map();
  }
  async findBestResponse(initialState: DialogueState): Promise<string> {
    const root = new MCTSNode(initialState);
    this.transpositionTable.set(root.state.hash(), root);

    const inProgress = new Set<string>();

    const simulationPromises = Array(this.numSimulations)
      .fill(null)
      .map(async () => {
        const path = await this.select(root, inProgress);
        const leaf = path[path.length - 1];

        if (!this.isTerminal(leaf.state)) {
          const expandedNode = await this.expand(leaf);
          const value = await this.simulate(expandedNode);
          this.backpropagate(path, value);
        }
      });

    await Promise.all(simulationPromises);

    // Select the child with the most visits
    if (root.children.length === 0) return "";

    const bestChild = root.children.reduce((best, child) =>
      child.getVisits() > best.getVisits() ? child : best,
    );

    return bestChild.action || "";
  }

  private async select(
    node: MCTSNode,
    inProgress: Set<string>,
  ): Promise<MCTSNode[]> {
    const path: MCTSNode[] = [];
    let current = node;

    while (!this.isTerminal(current.state) && current.isFullyExpanded) {
      path.push(current);

      if (current.children.length === 0) break;

      // Apply virtual loss
      const nodeHash = current.state.hash();
      if (inProgress.has(nodeHash)) {
        continue;
      }
      inProgress.add(nodeHash);

      // Selection using PUCT
      current = current.children.reduce((best, child) => {
        const totalVisits = current.getVisits();
        const priorProb =
          this.actionPriors.get(child.action || "") ||
          1.0 / current.children.length;
        return child.getPUCTScore(totalVisits, priorProb) >
          best.getPUCTScore(
            totalVisits,
            this.actionPriors.get(best.action || "") ||
              1.0 / current.children.length,
          )
          ? child
          : best;
      });
    }

    path.push(current);
    return path;
  }

  private async expand(node: MCTSNode): Promise<MCTSNode> {
    if (this.isTerminal(node.state)) return node;

    const actions = await this.generateActions(node.state);
    if (actions.length === 0) {
      node.isFullyExpanded = true;
      return node;
    }

    // Calculate action priors using policy network (in this case, the LLM)
    const priors = await this.calculateActionPriors(node.state, actions);

    for (let i = 0; i < actions.length; i++) {
      const action = actions[i];
      const newState = node.state.withNewMessage("assistant", action);

      // Check transposition table
      const stateHash = newState.hash();
      let childNode = this.transpositionTable.get(stateHash);

      if (!childNode) {
        childNode = new MCTSNode(newState, node, action);
        this.transpositionTable.set(stateHash, childNode);
      }

      node.children.push(childNode);
      this.actionPriors.set(action, priors[i]);
    }

    node.isFullyExpanded = true;
    return node.children[0];
  }

  private async simulate(node: MCTSNode): Promise<number> {
    let currentState = node.state;
    let depth = 0;
    let value = 0;

    while (!this.isTerminal(currentState) && depth < this.simulationDepth) {
      const actions = await this.generateActions(currentState);
      if (actions.length === 0) break;

      // Progressive widening
      const numActions = Math.max(1, Math.floor(Math.sqrt(depth + 1)));
      const selectedActions = actions.slice(0, numActions);

      const randomAction =
        selectedActions[Math.floor(Math.random() * selectedActions.length)];
      currentState = currentState.withNewMessage("assistant", randomAction);

      // Evaluate intermediate states with discount factor
      const intermediateValue = await this.evaluate(currentState);
      value += intermediateValue * 0.95 ** depth;

      depth++;
    }

    // Normalize accumulated value
    return value / (depth || 1);
  }

  private backpropagate(path: MCTSNode[], value: number): void {
    for (const node of path.reverse()) {
      node.addVisit(value);

      // Update RAVE statistics
      if (this.options.useRAVE) {
        node.addRaveData(value);
      }
    }
  }

  private async generateActions(state: DialogueState): Promise<string[]> {
    try {
      const messages: CoreMessage[] = [
        ...state.conversationHistory,
        {
          role: "user",
          content: state.currentQuery,
        },
      ];

      const completions: string[] = [];
      for (let i = 0; i < this.options.maxChildren; i++) {
        const { text, usage } = await generateText({
          model: this.model,
          maxTokens: 4096,
          temperature: 0.8 + i * 0.1, // Increase temperature for diversity
          system: state.systemPrompt,
          messages,
        });

        this.completionTokens += usage.completionTokens;
        completions.push(text);
      }

      return completions;
    } catch (error) {
      console.error("Error generating actions:", error);
      return [];
    }
  }

  private async evaluate(state: DialogueState): Promise<number> {
    try {
      const messages: CoreMessage[] = [
        ...state.conversationHistory,
        {
          role: "user",
          content: `
            Evaluate this conversation on the following criteria:
            1. Coherence (0-1)
            2. Relevance (0-1)
            3. Engagement (0-1)
            Respond with three numbers separated by commas.
          `,
        },
      ];

      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 256,
        temperature: 0.1,
        system: state.systemPrompt,
        messages,
      });

      this.completionTokens += usage.completionTokens;

      const [coherence, relevance, engagement] = text
        .split(",")
        .map((n) => Math.max(0, Math.min(Number.parseFloat(n.trim()), 1)));

      state.metrics = { coherence, relevance, engagement };

      // Weighted average of metrics
      return coherence * 0.3 + relevance * 0.4 + engagement * 0.3;
    } catch (error) {
      console.error("Error evaluating state:", error);
      return 0.5;
    }
  }

  private isTerminal(state: DialogueState): boolean {
    return (
      state.depth >= this.options.maxDepth ||
      state.currentQuery.toLowerCase().includes("goodbye") ||
      state.currentQuery.toLowerCase().includes("thank you")
    );
  }

  private async calculateActionPriors(
    state: DialogueState,
    actions: string[],
  ): Promise<number[]> {
    try {
      const messages: CoreMessage[] = [
        ...state.conversationHistory,
        {
          role: "user",
          content: `Rate the following potential responses in terms of their appropriateness (0-1):${actions.map((a, i) => `\n${i + 1}. ${a}`).join("")}`,
        },
      ];

      const { text, usage } = await generateText({
        model: this.model,
        maxTokens: 256,
        temperature: 0.1,
        system: state.systemPrompt,
        messages,
      });

      this.completionTokens += usage.completionTokens;

      const scores = text
        .split("\n")
        .map((line) => Number.parseFloat(line.trim()))
        .filter((score) => !Number.isNaN(score));

      // Softmax normalization
      const expScores = scores.map((score) => Math.exp(score));
      const sum = expScores.reduce((a, b) => a + b, 0);
      return expScores.map((score) => score / sum);
    } catch (_error) {
      // Uniform distribution as fallback
      return Array(actions.length).fill(1 / actions.length);
    }
  }
}

export async function mcts({
  model,
  system = "",
  prompt,
  numSimulations = 10,
  simulationDepth = 5,
  options = {},
}: {
  model: LanguageModel;
  system?: string;
  prompt: string;
  numSimulations?: number;
  simulationDepth?: number;
  options?: {
    maxDepth?: number;
    maxChildren?: number;
    explorationConstant?: number;
    temperature?: number;
    useRAVE?: boolean;
  };
}): Promise<[string, number]> {
  const mctsOptions = {
    maxDepth: options.maxDepth ?? 10,
    maxChildren: options.maxChildren ?? 3,
    explorationConstant: options.explorationConstant ?? 1.5,
    temperature: options.temperature ?? 0.8,
    useRAVE: options.useRAVE ?? true,
  };

  const mctsInstance = new MCTS(
    model,
    simulationDepth,
    numSimulations,
    mctsOptions,
  );

  const initialState = DialogueState.create(system, [], prompt);

  try {
    const response = await mctsInstance.findBestResponse(initialState);
    return [response, mctsInstance.completionTokens];
  } catch (error) {
    console.error("MCTS search failed:", error);
    throw error;
  }
}
