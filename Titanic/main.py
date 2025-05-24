import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product

# 1. CARREGAR OS DADOS
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'train.csv')
data = pd.read_csv(csv_path)

# 2. TRATAR VALORES AUSENTES
data['Age'] = data['Age'].fillna(data['Age'].median())

# 3. PRÉ-PROCESSAMENTO
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 60, 100], labels=['Child', 'Adult', 'Elderly'])
data['FareGroup'] = pd.cut(data['Fare'], bins=[0, 50, 100, 600], labels=['Low', 'Medium', 'High'])
data['AgeGroup'] = data['AgeGroup'].fillna('Adult')
data['FareGroup'] = data['FareGroup'].fillna('Medium')

# 4. DEFINIR A REDE BAYESIANA
model = DiscreteBayesianNetwork([
    ('Sex', 'Survived'),
    ('Pclass', 'Survived'),
    ('AgeGroup', 'Survived'),
    ('FareGroup', 'Survived')
])

# 5. CRIAR OS CPDs

# Estados possíveis
sex_states = ['male', 'female']
pclass_states = sorted(data['Pclass'].unique())
agegroup_states = ['Child', 'Adult', 'Elderly']
faregroup_states = ['Low', 'Medium', 'High']

# CPD para Sex
sex_counts = data['Sex'].value_counts(normalize=True).reindex(sex_states, fill_value=0)
cpd_sex_marginal = TabularCPD(
    variable='Sex',
    variable_card=len(sex_states),
    values=[[sex_counts[s]] for s in sex_states]
)

# CPD para Pclass
pclass_counts = data['Pclass'].value_counts(normalize=True).sort_index()
cpd_pclass = TabularCPD(
    variable='Pclass',
    variable_card=len(pclass_states),
    values=[[pclass_counts.get(cls, 0)] for cls in pclass_states]
)

# CPD para AgeGroup
age_counts = data['AgeGroup'].value_counts(normalize=True).reindex(agegroup_states, fill_value=0)
cpd_agegroup = TabularCPD(
    variable='AgeGroup',
    variable_card=len(agegroup_states),
    values=[[age_counts[label]] for label in age_counts.index]
)

# CPD para FareGroup
fare_counts = data['FareGroup'].value_counts(normalize=True).reindex(faregroup_states, fill_value=0)
cpd_faregroup = TabularCPD(
    variable='FareGroup',
    variable_card=len(faregroup_states),
    values=[[fare_counts[label]] for label in fare_counts.index]
)

# CPD para Survived condicional aos pais
all_combinations = list(product(sex_states, pclass_states, agegroup_states, faregroup_states))
comb_counts = {(s, p, a, f): [0, 0] for (s, p, a, f) in all_combinations}

for _, row in data.iterrows():
    key = (row['Sex'], row['Pclass'], row['AgeGroup'], row['FareGroup'])
    survived = row['Survived']
    if pd.isnull(key[2]) or pd.isnull(key[3]):
        continue
    comb_counts[key][survived] += 1

survived_0 = []
survived_1 = []

for key in all_combinations:
    total = comb_counts[key][0] + comb_counts[key][1]
    if total == 0:
        survived_0.append(0.5)
        survived_1.append(0.5)
    else:
        survived_0.append(comb_counts[key][0] / total)
        survived_1.append(comb_counts[key][1] / total)

cpd_survived = TabularCPD(
    variable='Survived',
    variable_card=2,
    values=[survived_0, survived_1],
    evidence=['Sex', 'Pclass', 'AgeGroup', 'FareGroup'],
    evidence_card=[
        len(sex_states),
        len(pclass_states),
        len(agegroup_states),
        len(faregroup_states)
    ]
)

# Adicionar CPDs
model.add_cpds(cpd_sex_marginal, cpd_pclass, cpd_agegroup, cpd_faregroup, cpd_survived)

# Verificar modelo
model.check_model()

# 6. INFERÊNCIA
infer = VariableElimination(model)

# 🔄 Conversão de rótulo para índice (Sex: 'female' → 1)
sex_mapping = {s: i for i, s in enumerate(sex_states)}

# Fazer inferência com entrada legível
evidence = {'Sex': sex_mapping['female']}
result = infer.query(variables=['Survived'], evidence=evidence)

# Função para exibir probabilidades
def exibir_probabilidades(resultado, evidencia_nome=None, evidencia_valor=None):
    if evidencia_nome is not None and evidencia_valor is not None:
        print(f"\n\nProbabilidades condicionadas à evidência: {evidencia_nome} = {evidencia_valor}\n")

    print("+=============+=================+")
    print(f"| Not Survived(0): |     {resultado.values[0]*100:.2f}% |")
    print("+-------------+-----------------+")
    print(f"| Survived(1):     |     {resultado.values[1]*100:.2f}% |")
    print("+-------------+-----------------+\n")

exibir_probabilidades(result, evidencia_nome='Sex', evidencia_valor='female')
