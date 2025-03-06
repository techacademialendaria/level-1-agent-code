import { createOpenAI } from "@ai-sdk/openai";
import { createAppAuth } from "@octokit/auth-app";
import { Octokit } from "@octokit/rest";
import { generateText } from "ai";
import bodyParser from "body-parser";
import { config } from "dotenv";
import express from "express";
import { parseStringPromise } from "xml2js";


config({ path: ".env.local" });

interface FileChange {
  filename: string; // Name of the file (e.g., "src/index.js")
  patch: string; // The diff changes (the lines that were added/removed)
  status: string; // The status of the file (modified, added, removed, etc.)
  additions: number; // Number of lines added
  deletions: number; // Number of lines deleted
  content?: string; // The actual current content of the file (Base64-decoded)
}

interface CodeAnalysis {
  summary: string; // A short summary of the pull request changes
  fileAnalyses: {
    path: string; // The path to the file being discussed
    analysis: string; // The AI's analysis for that file
  }[];
  overallSuggestions: string[]; // High-level recommendations or suggestions
}

const APP_ID = process.env.GITHUB_APP_ID;
const PRIVATE_KEY = process.env.GITHUB_PRIVATE_KEY;
const INSTALLATION_ID = process.env.GITHUB_INSTALLATION_ID;

if (!APP_ID || !PRIVATE_KEY || !INSTALLATION_ID) {
  throw new Error(
    `Missing required environment variables:
    APP_ID: ${!!APP_ID}
    PRIVATE_KEY: ${!!PRIVATE_KEY}
    INSTALLATION_ID: ${!!INSTALLATION_ID}`,
  );
}

if (!process.env.OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY environment variable.");
}

const octokit = new Octokit({
  authStrategy: createAppAuth,
  auth: {
    appId: APP_ID,
    privateKey: PRIVATE_KEY,
    installationId: INSTALLATION_ID,
  },
});

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  compatibility: "strict",
});


async function getFileContent(
  owner: string,
  repo: string,
  path: string,
  ref: string,
): Promise<string | undefined> {
  try {
    const response = await octokit.repos.getContent({ owner, repo, path, ref });
    // The response might contain various data structures depending on the type of the file or folder.
    if (
      "content" in response.data &&
      typeof response.data.content === "string"
    ) {
      // Decode the file content from Base64 to a readable string.
      return Buffer.from(response.data.content, "base64").toString();
    }
    return undefined;
  } catch (error: any) {
    if (error.status === 404) {
      // If the file wasn't found at that reference, log it out and return undefined.
      console.log(`File ${path} not found at ref ${ref}`);
      return undefined;
    }
    // If it's another error, rethrow it.
    throw error;
  }
}


async function parseReviewXml(xmlText: string): Promise<CodeAnalysis> {
  try {
    // We find where the <review> tag starts and ends to parse just that section.
    const xmlStart = xmlText.indexOf("<review>");
    const xmlEnd = xmlText.indexOf("</review>") + "</review>".length;

    // If we can't find the tags, we provide a fallback message.
    if (xmlStart === -1 || xmlEnd === -1) {
      console.warn(
        "Could not locate <review> tags in the AI response. Returning fallback.",
      );
      return {
        summary: "AI analysis could not parse the response from the model.",
        fileAnalyses: [],
        overallSuggestions: [],
      };
    }

    // Extract the portion of the string that contains the review XML.
    const xmlResponse = xmlText.slice(xmlStart, xmlEnd);

    // We use the xml2js library to parse the XML into a JavaScript object.
    const parsed = await parseStringPromise(xmlResponse);

    // Check if the parsed structure has the fields we need.
    if (
      !parsed.review ||
      !parsed.review.summary ||
      !parsed.review.fileAnalyses ||
      !parsed.review.overallSuggestions
    ) {
      console.warn(
        "Parsed XML is missing required fields. Returning fallback.",
      );
      return {
        summary: "AI analysis returned incomplete or invalid XML structure.",
        fileAnalyses: [],
        overallSuggestions: [],
      };
    }

    // Transform the parsed data into our CodeAnalysis interface shape.
    return {
      summary: parsed.review.summary[0] ?? "",
      fileAnalyses: Array.isArray(parsed.review.fileAnalyses[0].file)
        ? parsed.review.fileAnalyses[0].file.map((file: any) => ({
            path: file.path?.[0] ?? "Unknown file",
            analysis: file.analysis?.[0] ?? "",
          }))
        : [],
      overallSuggestions: Array.isArray(
        parsed.review.overallSuggestions[0].suggestion,
      )
        ? parsed.review.overallSuggestions[0].suggestion.map(
            (s: any) => s || "",
          )
        : [],
    };
  } catch (err) {
    // In case of any error (e.g., malformed XML), log it and provide fallback data.
    console.error("Error parsing AI-generated XML:", err);
    return {
      summary: "We were unable to fully parse the AI-provided code analysis.",
      fileAnalyses: [],
      overallSuggestions: [],
    };
  }
}

async function analyzeCode(
  title: string,
  changedFiles: FileChange[],
  commitMessages: string[],
): Promise<CodeAnalysis> {
  // We build a prompt that asks the AI to return an XML-formatted code review for the provided changes.
  const prompt = `Você é um especialista em revisão de código focado em prevenção de problemas. Analise estas mudanças de PR considerando especialmente:
  1. Impacto no fluxo principal de atendimento (criação de agente → configuração → WhatsApp)
  2. Potenciais problemas de performance (especialmente com banco de dados e webhook)
  3. Riscos para experiência do usuário e estabilidade da plataforma 
  4. Compatibilidade com múltiplos modelos de IA e integrações externas

  Escreva sua análise em parágrafos claros e concisos. Não use blocos de código para texto regular.
  Formate sugestões como itens de lista em uma única linha. Responda em português brasileiro como Rick Sanchez, com seu vocabulário ácido e palavrões.

  Contexto do SuperAgents:
  - Fluxos críticos: integração WhatsApp (Evolution/QR e ZAPI), Google Calendar, modelos de IA
  - Problemas conhecidos: performance do DataStore, race conditions no webhook, tempo de resposta
  - Foco atual: estabilidade em vez de novas funcionalidades, experiência do usuário

  Context:
  PR Title: ${title}
  Commit Messages: 
  ${commitMessages.map((msg) => '- ' + msg).join('\\n')}
  Changed Files:
  ${changedFiles
    .map(
      (file) => `
  File: ${file.filename}
  Status: ${file.status}
  Diff:
  ${file.patch}
  Current Content:
  ${file.content || "N/A"}
  `,
    )
    .join("\\n---\\n")}

  Forneça sua revisão no seguinte formato XML:
  <review>
    <summary>Escreva um parágrafo resumindo as mudanças e possíveis impactos nos fluxos críticos</summary>
    <fluxoImpact>Avalie o impacto específico no fluxo principal de atendimento</fluxoImpact>
    <performanceImpact>Avalie potenciais problemas de performance</performanceImpact>
    <fileAnalyses>
      <file>
        <path>caminho do arquivo</path>
        <analysis>Análise em parágrafos regulares, não blocos de código</analysis>
        <riskLevel>Alto|Médio|Baixo - justifique brevemente</riskLevel>
      </file>
    </fileAnalyses>
    <overallSuggestions>
      <suggestion>Escreva cada sugestão em uma única linha</suggestion>
    </overallSuggestions>
    <testingRecommendations>Recomendações específicas para testar essas mudanças antes do deploy</testingRecommendations>
  </review>
  
  Em seguida. Analise o código deste Pull Request e gere um checklist detalhado de testes para validação.

Contexto: Este PR contém alterações visuais e funcionais em nossa aplicação web. Preciso de um checklist abrangente que me ajude a validar todas as mudanças de forma sistemática.

Por favor:
1. Agrupe os testes por área funcional ou tipo de componente
2. Inclua testes específicos para mudanças de UI em diferentes temas (claro/escuro)
3. Identifique testes de regressão necessários para garantir que funcionalidades existentes não foram afetadas
4. Destaque áreas de alto risco que precisam de atenção especial
5. Inclua testes de responsividade e compatibilidade entre navegadores quando relevante

Formato desejado:
- Use uma lista com checkboxes "[ ]" para cada item de teste
- Organize os testes em seções lógicas com títulos em markdown (## Seção)
- Mantenha as instruções claras e específicas
- Inclua uma seção de "Testes de Regressão" para funcionalidades críticas

Considere tanto testes funcionais quanto visuais em sua análise.`;

  try {
    // Generate text using our OpenAI instance with the specified model (o1-mini in this example).
    const { text } = await generateText({
      model: openai("o1"),
      prompt,
    });

    // Parse the text the AI returned to extract the structured review in XML.
    return await parseReviewXml(text);
  } catch (error) {
    // If something goes wrong (AI call fails or parsing fails), return a fallback response.
    console.error("Error generating or parsing AI analysis:", error);
    return {
      summary: "We were unable to analyze the code due to an internal error.",
      fileAnalyses: [],
      overallSuggestions: [],
    };
  }
}

async function postPlaceholderComment(
  owner: string,
  repo: string,
  pullNumber: number,
): Promise<number> {
  const { data } = await octokit.issues.createComment({
    owner,
    repo,
    issue_number: pullNumber,
    body: "Desarmando essa bomba que você jogou aqui pra gente",
  });
  return data.id;
}

async function updateCommentWithReview(
  owner: string,
  repo: string,
  commentId: number,
  analysis: CodeAnalysis,
) {
  // We'll format the AI's analysis into a markdown comment.
  const finalReviewBody = `# Tech Rick Lead

${analysis.summary.trim()}

${analysis.fileAnalyses
  .map(
    (file) => `## ${file.path}
${file.analysis
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .join("\n")}`,
  )
  .join("\n\n")}

## Sugestões de Melhoria
${analysis.overallSuggestions.map((suggestion) => `• ${suggestion.trim()}`).join("\n")}

`;

  // Use GitHub's API to update the existing comment with our final review.
  await octokit.issues.updateComment({
    owner,
    repo,
    comment_id: commentId,
    body: finalReviewBody,
  });
}

async function handlePullRequestOpened(payload: any) {
  // The payload object from GitHub contains info like repository owner, repo name, PR number, etc.
  const owner = payload.repository.owner.login;
  const repo = payload.repository.name;
  const pullNumber = payload.pull_request.number;
  const title = payload.pull_request.title;
  const headRef = payload.pull_request.head.sha;

  try {
    // Post a placeholder comment on the PR so the user knows the bot is working.
    const placeholderCommentId = await postPlaceholderComment(
      owner,
      repo,
      pullNumber,
    );

    // List the changed files in the PR.
    const filesRes = await octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: pullNumber,
    });

    // For each file, retrieve content if it isn't removed. Then build the FileChange structure.
    const changedFiles: FileChange[] = await Promise.all(
      filesRes.data.map(async (file) => {
        let content: string | undefined;
        if (file.status !== "removed") {
          try {
            content = await getFileContent(owner, repo, file.filename, headRef);
          } catch (error) {
            console.error(
              `Error retrieving content for ${file.filename}:`,
              error,
            );
          }
        }
        return {
          filename: file.filename,
          patch: file.patch || "",
          status: file.status,
          additions: file.additions,
          deletions: file.deletions,
          content,
        };
      }),
    );

    // We also get the commit messages in the PR, which can help inform the AI's analysis.
    const commitsRes = await octokit.pulls.listCommits({
      owner,
      repo,
      pull_number: pullNumber,
    });
    const commitMessages = commitsRes.data.map((c) => c.commit.message);

    // Call our analyzeCode function to get the AI's review of these changes.
    const analysis = await analyzeCode(title, changedFiles, commitMessages);

    // Update our placeholder comment with the full review.
    await updateCommentWithReview(owner, repo, placeholderCommentId, analysis);

    console.log(
      `Submitted code review for PR #${pullNumber} in ${owner}/${repo}`,
    );
  } catch (error) {
    console.error(
      `Failed to handle 'pull_request' opened event for PR #${pullNumber}`,
      error,
    );
  }
}


const app = express();


app.use(bodyParser.json());


app.get("/", (req, res) => {
  res.send("PR Review Bot is running");
});


app.post("/webhook", async (req, res) => {
  try {
    // The event type is in the header. We're primarily looking for "pull_request" events.
    const eventType = req.headers["x-github-event"];
    const payload = req.body;

    // For a "pull_request" event, if the action is "opened", we handle it with our function above.
    if (eventType === "pull_request" && payload.action === "opened") {
      await handlePullRequestOpened(payload);
    }

    res.status(200).send("OK");
  } catch (error) {
    console.error("Error processing webhook:", error);
    res.status(500).send("Internal Server Error");
  }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`PR Review Bot listening on port ${PORT}`);
});
