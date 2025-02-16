import {
  type EmbeddingModel,
  type LanguageModel,
  embedMany,
  generateText,
} from "ai";

// biome-ignore lint/style/useNamingConvention: <explanation>
class ECHOImplementation {
  private model: LanguageModel;
  private embeddingModel: EmbeddingModel<string>;
  tokens = 0;

  constructor(model: LanguageModel, embeddingModel: EmbeddingModel<string>) {
    this.model = model;
    this.embeddingModel = embeddingModel;
  }

  // Step 1: Question Clustering
  async clusterQuestions(questions: string[], numClusters: number) {
    this.tokens = 0;
    // Generate embeddings for all questions
    const result = await embedMany({
      model: this.embeddingModel,
      values: questions,
    });
    // Implement k-means clustering
    return this.kMeansClustering(result, numClusters);
  }

  // Step 2: Demonstration Sampling
  async sampleDemonstrations(
    clusters: Array<{
      centroid: number[];
      points: Array<{ embedding: number[]; text: string }>;
    }>,
  ) {
    const demonstrations: {
      question: { embedding: number[]; text: string };
      rationale: string;
    }[] = [];
    for (const cluster of clusters) {
      const sortedQuestions = this.sortClusterPoints(cluster);
      for (const question of sortedQuestions) {
        const rationale = await this.generateZeroShotCoT(question.text);
        if (await this.satisfiesSelectionCriteria(question.text, rationale)) {
          demonstrations.push({
            question,
            rationale,
          });
          break;
        }
      }
    }
    return demonstrations;
  }

  // Step 3: Demonstration Unification
  async unifyDemonstrations(
    demonstrations: Array<{
      question: { embedding: number[]; text: string };
      rationale: string;
    }>,
    numIterations: number,
    finalCount: number,
  ) {
    const currentDemonstrations = [...demonstrations];
    for (let i = 0; i < numIterations; i++) {
      for (let j = 0; j < currentDemonstrations.length; j++) {
        const othersShuffled = this.shuffleArray(
          currentDemonstrations.filter((_, index) => index !== j),
        );
        const context = this.createContext(othersShuffled);
        const newRationale = await this.generateWithContext(
          currentDemonstrations[j]?.question.text ?? "",
          context,
        );
        currentDemonstrations[j] = {
          ...currentDemonstrations[j],
          rationale: newRationale,
        };
      }
    }
    return currentDemonstrations.slice(0, finalCount);
  }

  // Helper methods
  private async generateZeroShotCoT(question: string) {
    const prompt = `Let's solve this step by step:\nQuestion: ${question}\nLet's approach this step by step:`;
    const { text, usage } = await generateText({
      model: this.model,
      prompt,
    });
    this.tokens += usage.totalTokens;
    return text;
  }

  private createContext(
    demonstrations: Array<{ question: { text: string }; rationale: string }>,
  ) {
    return demonstrations
      .map((d) => `Question: ${d.question.text}\nSolution: ${d.rationale}`)
      .join("\n\n");
  }

  private async generateWithContext(question: string, context: string) {
    const prompt = `Here are some examples of solving similar problems:\n\n${context}\n\nNow, let's solve this question:\n${question}\nLet's approach this step by step:`;
    const { text, usage } = await generateText({
      model: this.model,
      prompt,
    });
    this.tokens += usage.totalTokens;
    return text;
  }

  private kMeansClustering(
    questions: { values: string[]; embeddings: number[][] },
    k: number,
  ) {
    const points = zip(questions.values, questions.embeddings).map((q) => ({
      embedding: q[1],
      text: q[0],
    }));

    // Initialize k random centroids
    let centroids = Array.from({ length: k }, () => {
      const randomIndex = Math.floor(Math.random() * points.length);
      return [...points[randomIndex].embedding];
    });

    let clusters: {
      centroid: number[];
      points: {
        embedding: number[];
        text: string;
      }[];
    }[];
    let previousClusters: {
      centroid: number[];
      points: {
        embedding: number[];
        text: string;
      }[];
    }[] = [];
    const maxIterations = 100;
    let iteration = 0;

    do {
      // Assign points to nearest centroid
      clusters = centroids.map((centroid) => ({
        centroid,
        points: [],
      }));

      for (const point of points) {
        let minDistance = Number.POSITIVE_INFINITY;
        let closestClusterIndex = 0;

        centroids.forEach((centroid, index) => {
          const distance = this.euclideanDistance(point.embedding, centroid);
          if (distance < minDistance) {
            minDistance = distance;
            closestClusterIndex = index;
          }
        });

        clusters[closestClusterIndex]?.points.push(point);
      }

      // Update centroids
      centroids = clusters.map((cluster) => {
        if (cluster.points.length === 0) {
          return cluster.centroid;
        }

        const newCentroid = new Array(cluster.points[0]?.embedding.length).fill(
          0,
        );
        for (const point of cluster.points) {
          point.embedding.forEach((value, dim) => {
            newCentroid[dim] += value;
          });
        }
        return newCentroid.map((sum) => sum / cluster.points.length);
      });

      // Check convergence
      const hasConverged =
        previousClusters && this.clustersEqual(clusters, previousClusters);
      if (hasConverged || iteration >= maxIterations) {
        break;
      }

      previousClusters = JSON.parse(JSON.stringify(clusters));
      iteration++;
    } while (true);

    return clusters;
  }

  private async satisfiesSelectionCriteria(
    question: string,
    rationale: string,
  ) {
    const prompt = `You will be given a question and rationale. Based on your judgement does the rationale satisfy the question Respond with only 'yes' or 'no'. Do not include any other text in your response.\nQuestion: ${question}\nRationale: ${rationale}`;
    const { text, usage } = await generateText({
      model: this.model,
      prompt,
    });
    this.tokens += usage.totalTokens;
    return text === "yes";
  }
  private sortClusterPoints(cluster: {
    centroid: number[];
    points: Array<{ embedding: number[]; text: string }>;
  }): { embedding: number[]; text: string }[] {
    return [...cluster.points].sort(
      (a, b) =>
        this.euclideanDistance(a.embedding, cluster.centroid) -
        this.euclideanDistance(b.embedding, cluster.centroid),
    );
  }

  private euclideanDistance(a: number[], b: number[]) {
    return Math.sqrt(
      a.reduce((sum, value, i) => sum + (value - (b[i] ?? 0)) ** 2, 0),
    );
  }

  private shuffleArray<T>(array: T[]): T[] {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      // Since we know i and j are valid indices (j is always <= i),
      // we can use a temporary variable to make the swap type-safe
      const temp = newArray[i];
      if (temp !== undefined && newArray[j] !== undefined) {
        newArray[i] = newArray[j];
        newArray[j] = temp;
      }
    }
    return newArray;
  }

  private clustersEqual(
    c1: Array<{ centroid: number[] }>,
    c2: Array<{ centroid: number[] }>,
  ) {
    return c1.every((cluster, i) =>
      cluster.centroid.every(
        (value, j) => Math.abs(value - (c2[i]?.centroid[j] ?? 0)) < 0.0001,
      ),
    );
  }
}

function zip<T, U>(arr1: T[], arr2: U[]): [T, U][] {
  const length = Math.min(arr1.length, arr2.length);
  const result: [T, U][] = [];
  for (let i = 0; i < length; i++) {
    result.push([arr1[i], arr2[i]]);
  }
  return result;
}

export async function echo({
  model,
  prompt,
  embeddingModel,
}: {
  model: LanguageModel;
  prompt: string;
  system?: string;
  embeddingModel: EmbeddingModel<string>;
}): Promise<[string, number]> {
  const echo = new ECHOImplementation(model, embeddingModel);

  // Sample questions
  const questions = [prompt];

  // Execute ECHO process
  const clusters = await echo.clusterQuestions(questions, 3);
  const initialDemonstrations = await echo.sampleDemonstrations(clusters);
  const finalDemonstrations = await echo.unifyDemonstrations(
    initialDemonstrations,
    5,
    2,
  );

  return [
    finalDemonstrations.map((demo) => demo.rationale).join("\n"),
    echo.tokens,
  ];
}
