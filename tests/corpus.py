"""Small synthetic bilingual corpus (EN/PT) used by tests and the calibration script.

Written for this project; no real book content. Each topic has two pages per language, so a
correct hit must come from the right document AND point at the right page.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class Topic(NamedTuple):
    category: str
    topics: List[str]
    en: List[str]
    pt: List[str]


CORPUS: Dict[str, Topic] = {
    "terraform": Topic(
        "iac-automation", ["terraform"],
        en=[
            "Terraform modules package related infrastructure resources into a reusable unit. A root module "
            "calls child modules and passes input variables, while outputs expose values back to the caller. "
            "Reusing modules keeps infrastructure code consistent across teams and environments.",
            "Remote state stored in an S3 bucket with locking prevents two engineers from applying changes at "
            "the same time. Running terraform plan shows the difference between the desired configuration and "
            "the recorded state before anything changes. Workspaces manage staging and production separately.",
        ],
        pt=[
            "Módulos do Terraform agrupam recursos de infraestrutura relacionados em uma unidade reutilizável. "
            "Um módulo raiz chama módulos filhos e passa variáveis de entrada, enquanto as saídas devolvem "
            "valores ao chamador. Reutilizar módulos mantém o código de infraestrutura consistente entre equipes.",
            "O estado remoto guardado em um bucket S3 com bloqueio impede que duas pessoas apliquem mudanças ao "
            "mesmo tempo. Executar terraform plan mostra a diferença entre a configuração desejada e o estado "
            "registrado antes de qualquer alteração. Workspaces separam homologação e produção.",
        ],
    ),
    "kubernetes": Topic(
        "containers-kubernetes", ["kubernetes"],
        en=[
            "A Kubernetes pod is the smallest deployable unit and runs one or more containers that share a "
            "network namespace. The scheduler assigns each pod to a node based on resource requests, while the "
            "kubelet on every node starts the containers and reports their status to the API server.",
            "A Deployment manages replica sets and performs rolling updates, replacing pods gradually so the "
            "application stays available. A Service gives a stable virtual address to a changing set of pods "
            "and balances traffic between them.",
        ],
        pt=[
            "Um pod do Kubernetes é a menor unidade implantável e executa um ou mais contêineres que "
            "compartilham o mesmo namespace de rede. O agendador atribui cada pod a um nó com base nos pedidos "
            "de recursos, e o kubelet de cada nó inicia os contêineres e informa seu estado ao servidor da API.",
            "Um Deployment gerencia conjuntos de réplicas e faz atualizações graduais, substituindo os pods aos "
            "poucos para que a aplicação continue disponível. Um Service dá um endereço virtual estável a um "
            "conjunto variável de pods e distribui o tráfego entre eles.",
        ],
    ),
    "kafka": Topic(
        "databases-messaging", ["kafka"],
        en=[
            "A Kafka topic is split into partitions, and each partition is an ordered, append-only log stored on "
            "a broker. Producers choose the partition by key, which keeps related events in order. Replication "
            "copies every partition to other brokers so a failure does not lose messages.",
            "Consumers in the same consumer group share the partitions of a topic, and each partition is read by "
            "only one consumer of the group at a time. Committed offsets record how far the group has read, so "
            "a restarted consumer resumes where it stopped.",
        ],
        pt=[
            "Um tópico do Kafka é dividido em partições, e cada partição é um log ordenado, apenas de "
            "acréscimo, guardado em um broker. Os produtores escolhem a partição pela chave, o que mantém "
            "eventos relacionados em ordem. A replicação copia cada partição para outros brokers para que uma "
            "falha não perca mensagens.",
            "Consumidores do mesmo grupo de consumidores dividem as partições de um tópico, e cada partição é "
            "lida por apenas um consumidor do grupo por vez. Os offsets confirmados registram até onde o grupo "
            "leu, então um consumidor reiniciado retoma de onde parou.",
        ],
    ),
    "postgres": Topic(
        "databases-messaging", ["postgresql"],
        en=[
            "A PostgreSQL index such as a B-tree lets the query planner find rows without scanning the whole "
            "table. Running EXPLAIN ANALYZE shows whether a slow query uses an index or a sequential scan, and "
            "how many rows the planner expected compared with the real count.",
            "PostgreSQL uses multiversion concurrency control, so readers do not block writers. Dead row "
            "versions accumulate until VACUUM removes them, and the write-ahead log makes committed changes "
            "durable after a crash.",
        ],
        pt=[
            "Um índice do PostgreSQL, como o B-tree, permite que o planejador de consultas encontre linhas sem "
            "varrer a tabela inteira. Executar EXPLAIN ANALYZE mostra se uma consulta lenta usa um índice ou "
            "uma varredura sequencial, e quantas linhas o planejador esperava em comparação com o total real.",
            "O PostgreSQL usa controle de concorrência multiversão, então leitores não bloqueiam escritores. "
            "Versões antigas de linhas se acumulam até que o VACUUM as remova, e o log de escrita antecipada "
            "torna as mudanças confirmadas duráveis após uma falha.",
        ],
    ),
    "sre": Topic(
        "devops-sre", ["slo"],
        en=[
            "A service level objective sets a target for a service level indicator, such as 99.9 percent of "
            "requests succeeding over thirty days. The error budget is the allowed amount of failure, and it "
            "tells the team how much risk it can take with new releases.",
            "Alerts based on the burn rate of the error budget page people only when users are really affected. "
            "Blameless postmortems record what happened and what to change, and reducing toil frees time for "
            "engineering work.",
        ],
        pt=[
            "Um objetivo de nível de serviço define uma meta para um indicador de nível de serviço, como 99,9 "
            "por cento das requisições com sucesso em trinta dias. O orçamento de erro é a quantidade de falha "
            "permitida e diz à equipe quanto risco ela pode assumir com novas versões.",
            "Alertas baseados na taxa de consumo do orçamento de erro acionam pessoas apenas quando os usuários "
            "são realmente afetados. Postmortems sem culpados registram o que aconteceu e o que mudar, e "
            "reduzir trabalho repetitivo libera tempo para engenharia.",
        ],
    ),
}


class Query(NamedTuple):
    text: str
    lang: str
    topic: str


QUERIES: List[Query] = [
    Query("How do I reuse infrastructure code with Terraform modules?", "en", "terraform"),
    Query("What does the kubelet do on each node?", "en", "kubernetes"),
    Query("How are Kafka partitions assigned to the consumers of a group?", "en", "kafka"),
    Query("Why is my PostgreSQL query slow and how do indexes help?", "en", "postgres"),
    Query("How does an error budget relate to an SLO?", "en", "sre"),
    Query("Como reutilizar código de infraestrutura com módulos do Terraform?", "pt", "terraform"),
    Query("O que o kubelet faz em cada nó?", "pt", "kubernetes"),
    Query("Como as partições do Kafka são atribuídas aos consumidores de um grupo?", "pt", "kafka"),
    Query("Por que minha consulta no PostgreSQL está lenta e como os índices ajudam?", "pt", "postgres"),
    Query("Qual é a relação entre orçamento de erro e SLO?", "pt", "sre"),
]

OFF_TOPIC: List[str] = [
    "What is a good recipe for chocolate cake?",
    "Who won the football World Cup in 1994?",
    "What is the capital city of Australia?",
    "Qual é uma boa receita de bolo de chocolate?",
    "Quem ganhou a Copa do Mundo de futebol em 1994?",
    "Qual é a capital da Austrália?",
]
