import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createAppAuth } from "@octokit/auth-app";
import { Octokit } from "@octokit/rest";
import { generateText } from "ai";
import bodyParser from "body-parser";
import { config } from "dotenv";
import express from "express";
import { parseStringPromise } from "xml2js";

config({ path: ".env.local" });

interface FileChange {
  filename: string; // Nome do arquivo (ex.: "src/index.js")
  patch: string; // As alterações em diff (linhas adicionadas/removidas)
  status: string; // Status do arquivo (modificado, adicionado, removido, etc.)
  additions: number; // Número de linhas adicionadas
  deletions: number; // Número de linhas removidas
  content?: string; // Conteúdo atual do arquivo (decodificado de Base64)
}

interface CodeAnalysis {
  summary: string; // Um breve resumo das alterações do pull request
  fluxoImpact: string; // Impacto no fluxo principal
  performanceImpact: string; // Impacto na performance
  fileAnalyses: {
    path: string; // Caminho do arquivo analisado
    analysis: string; // Análise da IA para esse arquivo
    riskLevel: string; // Nível de risco (Alto, Médio, Baixo)
  }[];
  overallSuggestions: string[]; // Recomendações ou sugestões de alto nível
  testingRecommendations: string; // Recomendações de testes
  testChecklist: string; // Checklist de testes formatado em markdown
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
const anthropic = createAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function getFileContent(
  owner: string,
  repo: string,
  path: string,
  ref: string,
): Promise<string | undefined> {
  try {
    const response = await octokit.repos.getContent({ owner, repo, path, ref });
    // A resposta pode conter várias estruturas de dados dependendo do tipo de arquivo ou pasta
    if (
      "content" in response.data &&
      typeof response.data.content === "string"
    ) {
      // Decodifica o conteúdo do arquivo de Base64 para uma string legível
      return Buffer.from(response.data.content, "base64").toString();
    }
    return undefined;
  } catch (error: any) {
    if (error.status === 404) {
      // Se o arquivo não foi encontrado nessa referência, registre e retorne undefined
      console.log(`File ${path} not found at ref ${ref}`);
      return undefined;
    }
    // Se for outro erro, relance-o
    throw error;
  }
}

async function parseReviewXml(xmlText: string): Promise<CodeAnalysis> {
  try {
    // Encontramos onde a tag <review> começa e termina para analisar apenas essa seção
    const xmlStart = xmlText.indexOf("<review>");
    const xmlEnd = xmlText.indexOf("</review>") + "</review>".length;

    // Se não conseguirmos encontrar as tags, fornecemos uma mensagem de fallback
    if (xmlStart === -1 || xmlEnd === -1) {
      console.warn(
        "Could not locate <review> tags in the AI response. Returning fallback.",
      );
      return {
        summary: "AI analysis could not parse the response from the model.",
        fluxoImpact: "Não foi possível analisar o impacto no fluxo.",
        performanceImpact:
          "Não foi possível analisar o impacto na performance.",
        fileAnalyses: [],
        overallSuggestions: [],
        testingRecommendations:
          "Não foi possível analisar as recomendações de teste.",
        testChecklist: "Não foi possível gerar o checklist de testes.",
      };
    }

    // Extraímos a parte da string que contém o XML de revisão
    const xmlResponse = xmlText.slice(xmlStart, xmlEnd);

    // Usamos a biblioteca xml2js para converter o XML em um objeto JavaScript
    const parsed = await parseStringPromise(xmlResponse);

    // Verificamos se a estrutura analisada tem os campos necessários
    if (!parsed.review) {
      console.warn(
        "Parsed XML is missing required fields. Returning fallback.",
      );
      return {
        summary: "AI analysis returned incomplete or invalid XML structure.",
        fluxoImpact: "Não foi possível analisar o impacto no fluxo.",
        performanceImpact:
          "Não foi possível analisar o impacto na performance.",
        fileAnalyses: [],
        overallSuggestions: [],
        testingRecommendations:
          "Não foi possível analisar as recomendações de teste.",
        testChecklist: "Não foi possível gerar o checklist de testes.",
      };
    }

    // Transformamos os dados analisados para o formato da nossa interface CodeAnalysis
    return {
      summary: parsed.review.summary?.[0] ?? "",
      fluxoImpact: parsed.review.fluxoImpact?.[0] ?? "",
      performanceImpact: parsed.review.performanceImpact?.[0] ?? "",
      fileAnalyses: Array.isArray(parsed.review.fileAnalyses?.[0]?.file)
        ? parsed.review.fileAnalyses[0].file.map((file: any) => ({
            path: file.path?.[0] ?? "Unknown file",
            analysis: file.analysis?.[0] ?? "",
            riskLevel: file.riskLevel?.[0] ?? "Médio",
          }))
        : [],
      overallSuggestions: Array.isArray(
        parsed.review.overallSuggestions?.[0]?.suggestion,
      )
        ? parsed.review.overallSuggestions[0].suggestion.map(
            (s: any) => s || "",
          )
        : [],
      testingRecommendations: parsed.review.testingRecommendations?.[0] ?? "",
      testChecklist: parsed.review.testChecklist?.[0] ?? "",
    };
  } catch (err) {
    // Em caso de erro (por exemplo, XML malformado), registre-o e forneça dados de fallback
    console.error("Error parsing AI-generated XML:", err);
    return {
      summary: "We were unable to fully parse the AI-provided code analysis.",
      fluxoImpact: "Não foi possível analisar o impacto no fluxo.",
      performanceImpact: "Não foi possível analisar o impacto na performance.",
      fileAnalyses: [],
      overallSuggestions: [],
      testingRecommendations:
        "Não foi possível analisar as recomendações de teste.",
      testChecklist: "Não foi possível gerar o checklist de testes.",
    };
  }
}

async function analyzeCode(
  title: string,
  changedFiles: FileChange[],
  commitMessages: string[],
): Promise<CodeAnalysis> {
  // Construímos um prompt que pede à IA para retornar uma revisão de código formatada em XML
  const prompt = `Você é um especialista em revisão de código focado em prevenção de problemas. Analise estas mudanças de PR considerando especialmente:
  1. Impacto no fluxo principal de atendimento (criação de agente → configuração → WhatsApp)
  2. Potenciais problemas de performance (especialmente com banco de dados e webhook)
  3. Riscos para experiência do usuário e estabilidade da plataforma 
  4. Compatibilidade com múltiplos modelos de IA e integrações externas

  Escreva sua análise em parágrafos claros e concisos. Não use blocos de código para texto regular.
  Formate sugestões como itens de lista em uma única linha. Responda em português brasileiro como Rick Sanchez, com seu vocabulário ácido e palavrões.
  <IMPORTANTE>
  Tenha muito cuidado com a formatação do XML. Ele deve ser formatado corretamente, sem erros e com a estrutura correta.
  <IMPORTANTE/>
  Contexto do SuperAgents:
  - Fluxos críticos: integração WhatsApp (Evolution/QR e ZAPI), Google Calendar, modelos de IA
  - Problemas conhecidos: performance do DataStore, race conditions no webhook, tempo de resposta
  - Foco atual: estabilidade em vez de novas funcionalidades, experiência do usuário

  Context:
  PR Title: ${title}
  Commit Messages: 
  ${commitMessages.map((msg) => "- " + msg).join("\\n")}
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
        <riskLevel>Alto|Médio|Baixo use emojis de semáforo junto para verde baixo, amarelo médio, vermelho alto - justifique brevemente</riskLevel>
      </file>
    </fileAnalyses>
    <overallSuggestions>
      <suggestion>Escreva cada sugestão em uma única linha</suggestion>
    </overallSuggestions>
    <testingRecommendations>Recomendações específicas para testar essas mudanças antes do deploy</testingRecommendations>
    <testChecklist>
    Adicione aqui um checklist completo de testes em formato markdown. Inclua:
    - Testes por área funcional/componente
    - Testes para UI em temas claro/escuro
    - Testes de regressão para funcionalidades existentes
    - Áreas de alto risco que precisam de atenção especial
    - Testes de responsividade/compatibilidade

    Use checkboxes "[ ]" para cada item e organize em seções com títulos "## Seção"
    </testChecklist>
  </review>
  `;

  try {
    // Geramos texto usando nossa instância do Anthropic (Claude)
    const { text } = await generateText({
      model: anthropic("claude-3-7-sonnet-latest"),
      prompt,
    });

    // Analisamos o texto retornado pela IA para extrair a revisão estruturada em XML
    return await parseReviewXml(text);
  } catch (error) {
    // Se algo der errado (chamada à IA falha ou análise falha), retornamos uma resposta de fallback
    console.error("Error generating or parsing AI analysis:", error);
    return {
      summary: "We were unable to analyze the code due to an internal error.",
      fluxoImpact: "Não foi possível analisar o impacto no fluxo.",
      performanceImpact: "Não foi possível analisar o impacto na performance.",
      fileAnalyses: [],
      overallSuggestions: [],
      testingRecommendations:
        "Não foi possível analisar as recomendações de teste.",
      testChecklist: "Não foi possível gerar o checklist de testes.",
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
    body: "Desarmando essa bomba que você jogou aqui pra gente!!",
  });
  return data.id;
}

async function updateCommentWithReview(
  owner: string,
  repo: string,
  commentId: number,
  analysis: CodeAnalysis,
) {
  // Formatamos a análise da IA em um comentário markdown
  const finalReviewBody = `# Tech Rick Lead

${analysis.summary.trim()}

## Impacto no Fluxo Principal
${analysis.fluxoImpact.trim()}

## Impacto na Performance
${analysis.performanceImpact.trim()}

${analysis.fileAnalyses
  .map(
    (file) => `## ${file.path} (Risco: ${file.riskLevel})
${file.analysis
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .join("\n")}`,
  )
  .join("\n\n")}

## Sugestões de Melhoria
${analysis.overallSuggestions.map((suggestion) => `• ${suggestion.trim()}`).join("\n")}

## Recomendações de Teste
${analysis.testingRecommendations.trim()}

# Checklist de Testes
${analysis.testChecklist.trim()}
`;

  // Usamos a API do GitHub para atualizar o comentário existente com nossa revisão final
  await octokit.issues.updateComment({
    owner,
    repo,
    comment_id: commentId,
    body: finalReviewBody,
  });
}

async function handlePullRequestOpened(payload: any) {
  // O objeto payload do GitHub contém informações como proprietário do repositório, nome do repo, número do PR, etc.
  const owner = payload.repository.owner.login;
  const repo = payload.repository.name;
  const pullNumber = payload.pull_request.number;
  const title = payload.pull_request.title;
  const headRef = payload.pull_request.head.sha;

  try {
    // Postamos um comentário placeholder no PR para que o usuário saiba que o bot está trabalhando
    const placeholderCommentId = await postPlaceholderComment(
      owner,
      repo,
      pullNumber,
    );

    // Listamos os arquivos alterados no PR
    const filesRes = await octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: pullNumber,
    });

    // Para cada arquivo, recuperamos o conteúdo se ele não foi removido. Em seguida, construímos a estrutura FileChange
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

    // Também obtemos as mensagens de commit no PR, o que pode ajudar a informar a análise da IA
    const commitsRes = await octokit.pulls.listCommits({
      owner,
      repo,
      pull_number: pullNumber,
    });
    const commitMessages = commitsRes.data.map((c) => c.commit.message);

    // Chamamos nossa função analyzeCode para obter a revisão da IA dessas alterações
    const analysis = await analyzeCode(title, changedFiles, commitMessages);

    // Atualizamos nosso comentário placeholder com a revisão completa
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
    // O tipo de evento está no cabeçalho. Estamos procurando principalmente eventos do tipo "pull_request"
    const eventType = req.headers["x-github-event"];
    const payload = req.body;

    // Para um evento "pull_request", se a ação for "opened", lidamos com ele usando nossa função acima
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
