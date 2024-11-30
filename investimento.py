import pandas as pd
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Previsão para Investimento de Clientes",
    layout="wide"
)

@st.cache_data
def load_model():
    dados = pd.read_csv('marketing_investimento.csv', sep=',')
    x = dados.drop('aderencia_investimento', axis=1)  # variáveis explicativas
    y = dados['aderencia_investimento']
    colunas = x.columns

    one_hot = make_column_transformer(
        (OneHotEncoder(drop='if_binary', handle_unknown='ignore'), ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
        remainder='passthrough', sparse_threshold=0
    )

    x = one_hot.fit_transform(x)
    pd.DataFrame(x, columns=one_hot.get_feature_names_out(colunas))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, stratify=y, random_state=5)

    normalizacao = MinMaxScaler()
    x_treino_normalizado = normalizacao.fit_transform(x_treino)
    pd.DataFrame(x_treino_normalizado)

    knn = KNeighborsClassifier()
    knn.fit(x_treino_normalizado, y_treino)

    x_teste_normalizado = normalizacao.transform(x_teste)
    knn.score(x_teste_normalizado, y_teste)

    filename = 'knn_model.pkl'
    pickle.dump(knn, open(filename, 'wb'))

    return dados, one_hot, normalizacao, knn

# Carregar o modelo e transformadores
dados, one_hot, normalizacao, knn_model = load_model()

# Streamlit app
st.title('Investimento Prediction App')

# Campos de entrada do usuário
idade = st.number_input('Idade', min_value=18, max_value=100, value=30)
saldo = st.number_input('Saldo', min_value=0, value=1500)
tempo_ult_contato = st.number_input('Tempo Último Contato', min_value=0, value=10)
numero_contatos = st.number_input('Número de Contatos', min_value=0, value=3)
estado_civil = st.selectbox('Estado Civil', ['solteiro', 'casado', 'divorciado'])
escolaridade = st.selectbox('Escolaridade', ['secundario', 'terciario', 'primario'])
inadimplencia = st.selectbox('Inadimplência', ['nao', 'sim'])
fez_emprestimo = st.selectbox('Fez Empréstimo', ['nao', 'sim'])

# Criar um DataFrame a partir da entrada do usuário
input_data = pd.DataFrame({
    'idade': [idade],
    'saldo': [saldo],
    'tempo_ult_contato': [tempo_ult_contato],
    'numero_contatos': [numero_contatos],
    'estado_civil': [estado_civil],
    'escolaridade': [escolaridade],
    'inadimplencia': [inadimplencia],
    'fez_emprestimo': [fez_emprestimo]
})

# Adicionar um botão para fazer a previsão
if st.button('Fazer Previsão'):
    # Transformar e normalizar a entrada do usuário
    transformed_input = one_hot.transform(input_data)
    transformed_input_normalized = normalizacao.transform(transformed_input)

    # Fazer a previsão
    prediction = knn_model.predict(transformed_input_normalized)

    # Exibir a previsão
    if prediction[0] == 1:
        st.write('Aderência ao Investimento: Sim')
    else:
        st.write('Aderência ao Investimento: Não')

# Sessão separada para gráficos
st.dataframe(dados)
st.title('Análise de Dados')

# Gráficos de histograma e boxplot
fig1 = px.histogram(dados, x='estado_civil', text_auto=True, color='aderencia_investimento', barmode='group')
fig2 = px.histogram(dados, x='escolaridade', text_auto=True, color='aderencia_investimento', barmode='group')
fig3 = px.histogram(dados, x='fez_emprestimo', text_auto=True, color='aderencia_investimento', barmode='group')
fig4 = px.box(dados, x='idade', color='aderencia_investimento')
fig5 = px.box(dados, x='saldo', color='aderencia_investimento')
fig6 = px.box(dados, x='tempo_ult_contato', color='aderencia_investimento')
fig7 = px.box(dados, x='numero_contatos', color='aderencia_investimento')

# Exibir gráficos no Streamlit
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.plotly_chart(fig5)
st.plotly_chart(fig6)
st.plotly_chart(fig7)