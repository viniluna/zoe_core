import os
import re
import streamlit as st
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import pandas as pd
from fuzzywuzzy import process

# Definição da estrutura de uma mensagem
@dataclass
class Mensagem:
    ator: str
    conteudo: str

# Definição de constantes para facilitar o uso
USUARIO = "usuario"
ASSISTENTE = "ai"
MENSAGENS = "mensagens"

# Função para obter o modelo LLM
@st.cache_resource
def obter_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name='gpt-3.5-turbo',
        openai_api_key='sk-a_clhEM-Ba75n3miFQcBGoaMGUpR46AgDb1dxI6NwDT3BlbkFJrgb_zL0sJ1P2X7X9fxVhs1inuFYT1dOokjdssVfXIA'
    )

# Função para criar o LLMChain com memória de conversação
def obter_llm_chain():
    template = """
    Você é a Zoe, uma assistente especializado em ajudar os clientes da Web Continental.
    Seu foco principal é ajudar os clientes com suas dúvidas, reclamações e solicitações sobre produtos da Web Continental.
    Você possui um vasto conhecimento sobre todos os produtos vendidos pela Web Continental, incluindo suas características, descrições, nomes, e links para compra.

    Histórico da conversa:
    {historico_conversa}

    Pergunta: 
    {pergunta}

    Resposta: 
    """
    template_prompt = PromptTemplate.from_template(template)
    memoria = ConversationBufferMemory(memory_key="historico_conversa")
    chain = LLMChain(
        llm=obter_llm(),
        prompt=template_prompt,
        verbose=True,
        memory=memoria
    )
    return chain

# Função para carregar dados dos produtos (exemplo para ar-condicionado)
def carregar_produtos():
    caminho_arquivo = r"./produtos/ar_condiconado.csv"
    produtos = pd.read_csv(caminho_arquivo)
    return produtos

# Função para listar todas as marcas disponíveis
def listar_marcas(df):
    return df['marca'].unique().tolist()

# Função para listar modelos disponíveis para uma marca específica
def listar_modelos_por_marca(df, marca):
    return df[df['marca'].str.lower() == marca.lower()]['modelo'].unique().tolist()

# Função para listar modelos e potências disponíveis para uma marca específica
def listar_modelos_e_potencias(df, marca):
    df_filtrado = df[df['marca'].str.lower() == marca.lower()]
    return df_filtrado[['modelo', 'potencia']].drop_duplicates().to_dict(orient='records')

# Função para listar todos os modelos disponíveis
def listar_todos_modelos(df):
    return df['modelo'].unique().tolist()

# Função para listar potência, marca e modelo juntos
def listar_potencia_marca_modelo(df):
    return df[['potencia', 'marca', 'modelo']].drop_duplicates().to_dict(orient='records')

# Função para determinar a intenção do usuário
def determinar_intencao(prompt):
    prompt_lower = prompt.lower()
    
    if "listar marcas" in prompt_lower:
        return "listar_marcas"
    elif "modelos da marca" in prompt_lower:
        marca = prompt_lower.replace("modelos da marca", "").strip()
        return ("listar_modelos_por_marca", marca)
    elif "modelos e potencias da marca" in prompt_lower:
        marca = prompt_lower.replace("modelos e potencias da marca", "").strip()
        return ("listar_modelos_e_potencias", marca)
    elif "listar todos os modelos" in prompt_lower:
        return "listar_todos_modelos"
    elif "listar potencia, marca e modelo" in prompt_lower:
        return "listar_potencia_marca_modelo"
    else:
        return "chatgpt"

# Função principal para processar a entrada do usuário e chamar a função correta
def processar_entrada(df, prompt):
    intencao = determinar_intencao(prompt)
    
    if intencao == "listar_marcas":
        return listar_marcas(df)
    elif isinstance(intencao, tuple) and intencao[0] == "listar_modelos_por_marca":
        return listar_modelos_por_marca(df, intencao[1])
    elif isinstance(intencao, tuple) and intencao[0] == "listar_modelos_e_potencias":
        return listar_modelos_e_potencias(df, intencao[1])
    elif intencao == "listar_todos_os_modelos":
        return listar_todos_modelos(df)
    elif intencao == "listar_potencia_marca_modelo":
        return listar_potencia_marca_modelo(df)
    elif intencao == "chatgpt":
        return responder_chatgpt(prompt)
    else:
        return "Desculpe, não entendi sua solicitação."

# Função para buscar informações sobre a Web Continental usando o LLM (GPT)
def responder_chatgpt(prompt):
    llm_chain = obter_llm_chain()
    resposta_gpt = llm_chain.run({"historico_conversa": "", "pergunta": prompt})
    return resposta_gpt

# Inicialização e execução do Streamlit
def inicializar_estado_sessao():
    if MENSAGENS not in st.session_state:
        st.session_state[MENSAGENS] = [
            Mensagem(
                ator=ASSISTENTE,
                conteudo="Olá, sou a Zoe sua especialista em produtos da Web Continental. Em que posso ajudar?"
            )
        ]
        
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = obter_llm_chain()
        
    if "produtos" not in st.session_state:
        st.session_state["produtos"] = carregar_produtos()  # Carrega os produtos do arquivo CSV

# Função para exibir a interface de chat no Streamlit
def exibir_chat():
    for msg in st.session_state[MENSAGENS]:
        st.chat_message(msg.ator).write(msg.conteudo)

    prompt = st.chat_input("Digite sua pergunta aqui")

    if prompt:
        st.session_state[MENSAGENS].append(Mensagem(ator=USUARIO, conteudo=prompt))
        st.chat_message(USUARIO).write(prompt)

        with st.spinner("Por favor, aguarde..."):
            resposta = processar_entrada(st.session_state["produtos"], prompt)
            st.session_state[MENSAGENS].append(Mensagem(ator=ASSISTENTE, conteudo=resposta))
            st.chat_message(ASSISTENTE).write(resposta)

# Função principal para rodar a aplicação
if __name__ == "__main__":
    inicializar_estado_sessao()
    exibir_chat()
