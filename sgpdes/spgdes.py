import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Perceptron
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
#from pure_sklearn.map import convert_estimator
import pickle
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier

## Funcionando bem
class SGPDES(BaseEstimator, ClassifierMixin):
    def __init__(self, WMA=25, ESD=0.001, EL=0.1, KI=7, max_iter=100000, pool_classifiers=None, DESNumbNN=5, Selector_Mode="MODELBASEDRF", CONSENSUSTH=95, resultprint=False):
        self.WMA = WMA
        self.ESD = ESD
        self.EL = EL
        self.KI = KI
        self.max_iter = max_iter
        #self.num_classifier_base = num_classifier_base
        self.DESNumbNN = DESNumbNN
        self.Selector_Mode = Selector_Mode
        self.CONSENSUSTH = CONSENSUSTH
        self.pool_classifiers_ = pool_classifiers
        self.pool_classifiers = pool_classifiers
        self.resultprint = resultprint

    def fit(self, X, y):
        # Valida os inputs
        X, y = check_X_y(X, y)

        # Criando o conjunto TR
        TR = np.column_stack((X, y))

        # Executar o HSGP para gerar os protótipos
        # Supõe-se que hsgp_tracking é uma função definida em outro lugar
        # As variáveis como WMA, ESD, etc. devem ser acessadas com self.WMA, self.ESD, etc.
        R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes = self._hsgp_tracking(TR, self.WMA, self.ESD, self.EL, self.KI, self.max_iter)

        # Armazena os dados do protótipo para uso posterior
        self.prototype_data_ = pd.DataFrame(np.array(R))

        # Gerar o Pool de Classificadores base
        # Supõe-se que poolGeneration é uma função definida em outro lugar
        # O uso de self.num_classifier_base para passar o número de classificadores base
        X = pd.DataFrame(X)
        y = pd.Series(y)
        # Dividindo o conjunto de dados em treinamento e teste
        #X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Extrair uma amostra estratificada de 10% para treinar o metaclassificador
        #X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.01, stratify=y, random_state=42)

        prototype_data = np.array(R)
        prototype_data = pd.DataFrame(prototype_data)
        # Cálculo do número de elementos em cada array
        num_TR = len(TR)
        num_R = len(R)

        # Cálculo da taxa de redução
        taxa_reducao = (num_TR - num_R) / num_TR
        #print("Taxa de Redução do DSEL:", taxa_reducao)
        # Testando a função de geração do pool de classificadores
        #num_classifier_base = 100
        #max_iter = 1000
        #pool_classifier_base = poolGeneration(X_train, X_test, Y_train, Y_test, num_classifier_base)
        #self.pool_classifiers_ = poolGeneration(X_train, X_test, Y_train, Y_test, self.num_classifier_base)

        # Treinamento do classificador com os protótipos e o pool de classificadores
        # Supõe-se que train_sgpdes é uma função definida em outro lugar
        # Nota: As funções externas devem ser adaptadas para aceitar e retornar os tipos de dados esperados
        self.clt_sgpdes_ = self._train_sgpdes(
            prototype_data=self.prototype_data_,
            X_train_meta=X,
            Y_train_meta=y,
            pool_classifiers=self.pool_classifiers_,
            DESNumbNN=self.DESNumbNN,
            Selector_Mode=self.Selector_Mode,
            resultprint = self.resultprint
        )

        return self,taxa_reducao

    def predict(self, X):
        # Verifica se fit já foi chamado
        check_is_fitted(self, ['prototype_data_', 'pool_classifiers_', 'clt_sgpdes_'])

        # Valida os inputs
        X = check_array(X)
        X = pd.DataFrame(X)

        # Lista para coletar as previsões de cada linha
        predictions = []

        # Loop através de cada linha de X_test usando itertuples para eficiência
        for i in range(len(X)):
            # Fazendo a previsão para a linha atual
            final_prediction = self._predict_sgpdes(
                prototype_data= self.prototype_data_,
                X_test=X.iloc[i:i+1,:],  # Selecionando apenas a linha atual
                pool_classifiers=self.pool_classifiers_,
                DESNumbNN=self.DESNumbNN,
                Selector_Mode=self.Selector_Mode,
                clt_sgpdes=self.clt_sgpdes_,
                CONSENSUSTH=self.CONSENSUSTH,
                resultprint=self.resultprint
            )
            # Adicionando a previsão à lista de previsões
            predictions.append(final_prediction)
        # Converter todos os elementos para escalares
        predictions = [x[0] if isinstance(x, np.ndarray) else x for x in predictions]

        # Implemente aqui a lógica de previsão usando o estado do objeto definido em fit
        # Supõe-se que predict_sgpdes é uma função definida em outro lugar
        #final_prediction = self._predict_sgpdes(
        #    prototype_data=self.prototype_data_,
        #    X_test=X,  # Selecionando apenas a linha atual
        #    pool_classifiers=self.pool_classifiers_,
        #    DESNumbNN=self.DESNumbNN,
        #    Selector_Mode=self.Selector_Mode,
        #    clt_sgpdes=self.clt_sgpdes_,
        #    CONSENSUSTH=self.CONSENSUSTH
        #)

        return predictions

    def score(self, X, y):

        check_is_fitted(self, ['prototype_data_', 'pool_classifiers_', 'clt_sgpdes_'])

        # Valida os inputs
        X = check_array(X)
        X = pd.DataFrame(X)

        # Lista para coletar as previsões de cada linha
        predictions = []


        for i in range(len(X)):
            # Fazendo a previsão para a linha atual
            final_prediction = self._predict_sgpdes(
            prototype_data= self.prototype_data_,
            X_test=X.iloc[i:i+1,:],  # Selecionando apenas a linha atual
            pool_classifiers=self.pool_classifiers_,
            DESNumbNN=self.DESNumbNN,
            Selector_Mode=self.Selector_Mode,
            clt_sgpdes=self.clt_sgpdes_,
            CONSENSUSTH=self.CONSENSUSTH,
            resultprint=self.resultprint
          )
            # Adicionando a previsão à lista de previsões
            predictions.append(final_prediction)
        # Converter todos os elementos para escalares
        predictions = [x[0] if isinstance(x, np.ndarray) else x for x in predictions]


        # Convertendo a lista de previsões em um array para calcular a acurácia
        predictions = np.array(predictions)
        y_test = np.array(y)
        #predictions = 1 - predictions
        # Calculando a acurácia comparando as previsões com os rótulos verdadeiros
        accuracy = accuracy_score(y_test, predictions)
        #print(f"Acurácia: {accuracy * 100.0}%")

        return accuracy

    def _train_sgpdes(self, prototype_data, X_train_meta, Y_train_meta, pool_classifiers, DESNumbNN, Selector_Mode, resultprint):
        """
        Função para treinar um classificador usando a abordagem SGP-DES baseada na complexidade dos dados e métricas de competência.

        :param prototype_data: Dados protótipo utilizados para o cálculo das métricas de complexidade.
        :param X_train_meta: Dados de treinamento.
        :param Y_train_meta: Rótulos de treinamento.
        :param complexity_metrics_df: DataFrame de métricas de complexidade (não utilizado nesta versão da função).
        :param pool_classifiers: Conjunto de classificadores a serem utilizados.
        :param DESNumbNN: Número de vizinhos mais próximos para o cálculo das métricas de complexidade.
        :param Selector_Mode: Modo do seletor a ser utilizado.
        :return: Um modelo treinado.
        """
        # Preparando os dados de complexidade para os exemplos de treinamento
        dsel_data_complexity_metrics = self._dsel_generation_complexity_metrics(prototype_data, X_train_meta, DESNumbNN, pool_classifiers, resultprint)
        dsel_data_complexity_metrics_competence = self._dsel_generation_complexity_metrics_competence(X_train_meta, Y_train_meta, dsel_data_complexity_metrics, self.pool_classifiers_)

        # Resetando os índices dos DataFrames antes de concatenar para evitar o InvalidIndexError
        dsel_data_complexity_metrics_reset = dsel_data_complexity_metrics.reset_index(drop=True)
        dsel_data_complexity_metrics_competence_reset = dsel_data_complexity_metrics_competence.reset_index(drop=True)

        # Concatenando os DataFrames de métricas de complexidade e competência
        dsel_data_complexity_combined = pd.concat([dsel_data_complexity_metrics_reset, dsel_data_complexity_metrics_competence_reset], axis=1)
        dsel_data_complexity_combined = dsel_data_complexity_combined.drop(columns='classifier_id')
        #print(dsel_data_complexity_combined)
        # Se o modo do seletor for baseado em modelo XGBoost
        if Selector_Mode == "MODELBASEDXGB":
            # Preparando os dados para o XGBoost
            dtrain = xgb.DMatrix(dsel_data_complexity_combined.iloc[:, :-1], label=dsel_data_complexity_combined.iloc[:, -1])

            # Definindo os parâmetros do modelo XGBoost
            params = {
                'max_depth': 3,  # Profundidade máxima de cada árvore
                'eta': 0.1,      # Taxa de aprendizado
                'objective': 'binary:logistic',  # Função objetivo para classificação binária
                'eval_metric': 'logloss'         # Métrica de avaliação
            }

            # Número de rodadas de treinamento
            num_rounds = 100

            # Treinando o modelo XGBoost
            clt_sgpdes = xgb.train(params, dtrain, num_rounds)
        # Verifica se o modo do seletor é baseado no modelo SVM
        if Selector_Mode == "MODELBASEDSVM":

           # Criação do pipeline SVM com pré-processamento
           # O uso de um StandardScaler é recomendado para SVM para normalizar os dados
           svm_pipeline = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='auto', probability=True))

           # Treinando o modelo SVM
           clt_sgpdes = svm_pipeline.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

         # Verifica se o modo do seletor é baseado no modelo KNN
        if Selector_Mode == "MODELBASEDKNN":
           # Criação do pipeline KNN com pré-processamento
           # Normalização dos dados é importante para o KNN devido à sua natureza sensível à distância
           knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

           # Treinando o modelo KNN
           # dsel_data_complexity_combined.iloc[:, :-1] são os dados de treinamento (features)
           # dsel_data_complexity_combined.iloc[:, -1] são os rótulos de treinamento
           clt_sgpdes = knn_pipeline.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

        # Verifica se o modo do seletor é baseado no modelo RandomForest
        if Selector_Mode == "MODELBASEDRF":
           # Criação do modelo RandomForest
           #print(dsel_data_complexity_combined)
           # A normalização dos dados não é estritamente necessária para modelos baseados em árvores
           rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

           # Treinando o modelo RandomForest
           # dsel_data_complexity_combined.iloc[:, :-1] são os dados de treinamento (features)
           # dsel_data_complexity_combined.iloc[:, -1] são os rótulos de treinamento
           clt_sgpdes = rf_model.fit(dsel_data_complexity_combined.iloc[:, :-1], dsel_data_complexity_combined.iloc[:, -1])

        return clt_sgpdes

    def _predict_sgpdes(self, prototype_data, X_test, pool_classifiers, DESNumbNN, Selector_Mode, clt_sgpdes, CONSENSUSTH, resultprint=False):
        #X_test = X_test[0:1]
        """
        Função para fazer previsões com um modelo SGP-DES.
        X_test
        :param prototype_data: Dados protótipo utilizados para o cálculo das métricas de complexidade.
        :param X_test: Dados de teste para fazer previsões.
        :param pool_classifiers: Conjunto de classificadores a serem utilizados.
        :param DESNumbNN: Número de vizinhos mais próximos para o cálculo das métricas de complexidade.
        :param Selector_Mode: Modo do seletor a ser utilizado.
        :param clt_sgpdes: Modelo treinado para fazer previsões.
        :return: Previsão final após a votação majoritária.
        """
        # Armazenar previsões de todos os classificadores para cada amostra
        all_predictions = np.array([classifier.predict(X_test) for classifier in pool_classifiers]).T
        predictions = []
        # Calcular o modo (valor mais frequente) para cada linha (amostra) para encontrar a previsão de consenso
        consensus_predictions, count = mode(all_predictions, axis=1, keepdims=True)

        # Calcula o percentual de consenso transformando a contagem em percentual
        consensus_percentage = count / len(pool_classifiers) * 100

        # Consensus_percentage agora contém o percentual de consenso para cada amostra
        overall_consensus_percentage = np.mean(consensus_percentage)

        # Se o consenso geral for maior que o Threshold, faz a previsão com todos os classificadores base
        if overall_consensus_percentage >= CONSENSUSTH:
           pred_scalar = mode(all_predictions, axis=None,keepdims=True)[0][0]
           #print("consenso")
           return pred_scalar

        # Gerando métricas de complexidade para os dados de teste
        dsel_data_complexity_metrics_test = self._dsel_generation_complexity_metrics(prototype_data, X_test, DESNumbNN, pool_classifiers, resultprint)

        if Selector_Mode == "MODELBASEDXGB":
            dtest = xgb.DMatrix(dsel_data_complexity_metrics_test)
            preds = clt_sgpdes.predict(dtest)
            predictions = np.where(preds > 0.5, 1, 0)

        if Selector_Mode == "MODELBASEDSVM":
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)
            #predictions = np.where(pred_probs > 0.5, 1, 0)

        if Selector_Mode == "MODELBASEDKNN":
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)
            # Convertendo probabilidades em previsões de classe
            #predictions = np.argmax(pred_probs, axis=1)
            #print("knn")

        if Selector_Mode == "MODELBASEDRF":
            #pred_probs = clt_sgpdes.predict_proba(dsel_data_complexity_metrics_test)[:, 1]
            #predictions = np.where(pred_probs > 0.5, 1, 0)
            predictions = clt_sgpdes.predict(dsel_data_complexity_metrics_test)
            #print(predictions)
        selected_predictions = []

        for idx, prediction1 in enumerate(predictions):
            if prediction1 == 0:
                pred = pool_classifiers.estimators_[idx].estimator.predict(X_test)
                selected_predictions.extend(pred)

        if not selected_predictions:
            pred_scalar = mode(all_predictions, axis=None,keepdims=True)[0][0]
            #print("consenso")
            return pred_scalar

        all_predictions = np.array(selected_predictions)
        global_majority_vote =  mode(all_predictions, axis=None,keepdims=True)[0][0]

        #print("vote")
        #print(global_majority_vote)

        return global_majority_vote


    def _dsel_generation_complexity_metrics(self, prototype_data, X_complexy, DESNumbNN, pool_classifiers, resultprint):
        import time
        #X_complexy = X_test
        X_prototype = prototype_data.iloc[:, :-1]
        y_prototype = prototype_data.iloc[:, -1]
        complexity_metrics_list = []
        # Declara um DataFrame vazio
        complexity_metrics_list_df = pd.DataFrame()
        complexity_metrics_df = pd.DataFrame()
        nbrs = NearestNeighbors(n_neighbors=DESNumbNN, algorithm='auto').fit(X_prototype)
        #x = 1
        # Inicializar listas para armazenar tempos de execução das partes específicas
        times_neighbors = []
        times_prototype_processing = []
        times_distance_calculation = []
        times_metric_calculation = []
        #x = 178
        for x in range(len(X_complexy)):
            start_neighbors = time.time()
            _, indices = nbrs.kneighbors([X_complexy.iloc[x, :]])
            end_neighbors = time.time()
            times_neighbors.append(end_neighbors - start_neighbors)

            start_prototype_processing = time.time()
            nearest_prototypes = X_prototype.iloc[indices[0], :]
            centroid = nearest_prototypes.mean().values.reshape(1, -1)
            centroid_full = centroid
            end_prototype_processing = time.time()
            times_prototype_processing.append(end_prototype_processing - start_prototype_processing)

            start_distance_calculation = time.time()
            distance_X_Centroid = distance.cdist([X_complexy.iloc[x, :]], centroid, 'euclidean')
            end_distance_calculation = time.time()
            times_distance_calculation.append(end_distance_calculation - start_distance_calculation)

            start_metric_calculation = time.time()
            metrics_int = self._scores_BaseClassifiersPYTHON5(
                X_localRegion=nearest_prototypes,
                y_localRegion=y_prototype.iloc[indices[0]],
                X_prev=X_complexy.iloc[x, :],
                n_classifier=len(pool_classifiers),
                pool_classifiers=pool_classifiers,
            )
            end_metric_calculation = time.time()
            times_metric_calculation.append(end_metric_calculation - start_metric_calculation)

            # Certifique-se de que metrics_int é um np.array; caso contrário, converta-o.
            metrics_int = np.array(metrics_int)
            centroid = np.repeat(distance_X_Centroid[:, [0]], len(metrics_int))

            # Concatenar horizontalmente o array de métricas com o valor de distância (transformado em array).
            #metrics_with_distance = np.hstack((metrics_int.reshape(1, -1), centroid))


            # Se distance_X_Centroid é para ser apenas uma coluna adicional
            df_distance_X_Centroid = pd.DataFrame(centroid, columns=['distance'])

             # Convertendo os arrays NumPy para DataFrames do pandas
            df_metrics_int = pd.DataFrame(metrics_int, columns=[f'metric_{i}' for i in range(metrics_int.shape[1])])

            # Concatenando sem repetição
            df_concatenado = pd.concat([df_metrics_int, df_distance_X_Centroid], axis=1)
            num_rows = df_concatenado.shape[0]
            # Convertendo o array centróide para um DataFrame
            # Repetindo o centróide n vezes usando np.tile
            centroid_repeated = np.tile(centroid_full, (num_rows, 1))
            centroid_df = pd.DataFrame(centroid_repeated)
            #print(type(centroid_full))
            df_concatenado = pd.concat([df_concatenado,centroid_df], axis=1)
            #print(df_concatenado)
            # Armazenando as métricas calculadas
            #complexity_metrics_list.append(np.append(df_concatenado))

            # Concatenando os dois DataFrames para formar um único DataFrame
            complexity_metrics_df = pd.concat([complexity_metrics_df, df_concatenado], axis=0)
            complexity_metrics_df = pd.DataFrame(complexity_metrics_df, columns=[f'metric_{i}' for i in range(metrics_int.shape[1])])


        # Calculando as médias dos tempos
        average_time_neighbors = sum(times_neighbors) / len(times_neighbors)
        average_time_prototype_processing = sum(times_prototype_processing) / len(times_prototype_processing)
        average_time_distance_calculation = sum(times_distance_calculation) / len(times_distance_calculation)
        average_time_metric_calculation = sum(times_metric_calculation) / len(times_metric_calculation)
        #print(centroid)
        # Imprimindo os resultados
        if resultprint == True:
            print(f"Tempo médio para busca dos vizinhos: {average_time_neighbors} segundos")
            print(f"Tempo médio para processamento dos protótipos: {average_time_prototype_processing} segundos")
            print(f"Tempo médio para cálculo de distância: {average_time_distance_calculation} segundos")
            print(f"Tempo médio para cálculo das métricas: {average_time_metric_calculation} segundos")
        # Convertendo a lista de métricas em um DataFrame
        #complexity_metrics_df.columns=['overall_support_local_region', 'condition_class_support_local_region', 'score_overall', 'score_class_condition_np', 'overall_class_condition_distance_decision_boundary', 'newSample_distance_decision_boundary', 'distance_X_Centroid', 'centroid']


        return complexity_metrics_df


    def _dsel_generation_complexity_metrics_competence(self, X_train, Y_train,complexity_metrics_df, pool_classifiers):
        # Inicializando listas para armazenar os resultados
        dsel_data_predict_list = []
        classifiers_id_list = []

        for x in range(len(X_train)):
            for m in range(len(pool_classifiers)):
                # Adicionando o resultado de competência e o ID do classificador às listas
                dsel_data_predict_list.append(1 - (pool_classifiers.estimators_[m].estimator.predict(X_train.iloc[x, :].values.reshape(1, -1)) == Y_train.iloc[x]).astype(int))
                classifiers_id_list.append(m)  # Adicionando 1 porque o índice em Python começa em 0

        # Convertendo listas para DataFrames pandas
        dsel_data_predict_df = pd.DataFrame(dsel_data_predict_list, columns=['competence'])
        classifiers_id_df = pd.DataFrame(classifiers_id_list, columns=['classifier_id'])

        # Combinando os resultados em um único DataFrame
        dsel_data_complexity_predict = pd.concat([classifiers_id_df, dsel_data_predict_df], axis=1)

        # Retornando o DataFrame combinado
        return dsel_data_complexity_predict

        import numpy as np

    def _scores_BaseClassifiersPYTHON5_optimized(self, X_localRegion, y_localRegion, X_prev, n_classifier, pool_classifiers):
        # Assume inputs are already np.ndarray for optimization
        X_localRegion = np.asarray(X_localRegion)
        y_localRegion = np.asarray(y_localRegion)
        X_prev_np = X_prev.values.reshape(1, -1)
        X_prev_np = np.atleast_2d(X_prev.values)

        # Preparing result matrix
        results = np.zeros((n_classifier, 6))

        # Computing decisions, scores, class predictions, and probabilities once per classifier
        decisions = [clf.estimator.decision_function(X_prev_np) for clf in pool_classifiers.estimators_]
        scores = [clf.estimator.score(X_localRegion, y_localRegion) for clf in pool_classifiers.estimators_]
        class_predictions = [clf.estimator.predict(X_prev_np) for clf in pool_classifiers.estimators_]
        prob_supports = [clf.predict_proba(X_localRegion) for clf in pool_classifiers.estimators_]
        max_probs = [prob.max(axis=1) for prob in prob_supports]
        overall_supports = [max_prob.mean() for max_prob in max_probs]

        # Vectorized handling of per-classifier metrics if possible
        for idx, clf in enumerate(pool_classifiers.estimators_):
            mask = y_localRegion == class_predictions[idx]
            relevant_data = X_localRegion[mask]

            # Calculating metrics for relevant data
            if relevant_data.size > 0:
                relevant_labels = y_localRegion[mask]
                class_score = clf.estimator.score(relevant_data, relevant_labels)
                class_decision = clf.estimator.decision_function(relevant_data)
                class_prob_support = clf.predict_proba(relevant_data).max(axis=1).mean()
                class_decision_mean = class_decision.mean()
            else:
                class_score = 0
                class_decision_mean = 0
                class_prob_support = 0

            # Updating results for each classifier
            results[idx] = [
                overall_supports[idx],
                class_prob_support,
                scores[idx],
                class_score,
                class_decision_mean,
                decisions[idx].mean()
            ]

        return results




    def _scores_BaseClassifiersPYTHON5(self, X_localRegion, y_localRegion, X_prev, n_classifier, pool_classifiers):
        X_localRegion = np.asarray(X_localRegion)
        y_localRegion = np.asarray(y_localRegion)
        X_prev_np = X_prev.values.reshape(1, -1)

        # Preparar matriz de resultados
        results = np.zeros((n_classifier, 6))

        # Obter as decisões e probabilidades para X_prev e X_localRegion fora do loop
        decisions = np.array([clf.estimator.decision_function(X_prev_np) for clf in pool_classifiers.estimators_[:n_classifier]])
        scores = np.array([clf.estimator.score(X_localRegion, y_localRegion) for clf in pool_classifiers.estimators_[:n_classifier]])
        class_predictions = np.array([clf.estimator.predict(X_prev_np) for clf in pool_classifiers.estimators_[:n_classifier]])

        # Probabilidades gerais
        prob_supports = np.array([clf.predict_proba(X_localRegion) for clf in pool_classifiers.estimators_[:n_classifier]])
        max_probs = prob_supports.max(axis=2)
        overall_supports = max_probs.mean(axis=1)

        # Processar cada classificador
        for idx in range(n_classifier):
            clf = pool_classifiers[idx]
            mask = y_localRegion == class_predictions[idx]
            relevant_data = X_localRegion[mask]

            # Calcular métricas para dados
            if relevant_data.size > 0:
                class_score = clf.estimator.score(relevant_data, y_localRegion[mask])
                class_decision = clf.estimator.decision_function(relevant_data)
                class_prob_support = clf.predict_proba(relevant_data).max(axis=1).mean()
                class_decision_mean = class_decision.mean()
            else:
                class_score = 0
                class_decision_mean = 0
                class_prob_support = 0

            results[idx] = [
                overall_supports[idx],
                class_prob_support,
                scores[idx],
                class_score,
                class_decision_mean,
                decisions[idx].mean()
            ]

        return results



    def _predict_BaseClassifiersPYTHON4(self,X_prev,n_classifier,pool_classifiers):

        score_all_prev = np.zeros((0), float)
        score_all_prev_np = np.zeros((0), float)
        #x = 1


        for x in range(int(n_classifier)):
          #print("error aqui!")
          score_all_prev = pool_classifiers.estimators_[x].estimator.predict(X_prev)
          score_all_prev_np = np.append(score_all_prev_np, np.array(score_all_prev), axis=None)


        score_all_prev = score_all_prev_np
        #vctr12 = scoreBaseEstimator_all
        return score_all_prev

    def _hsgp_tracking(self,TR, WMA, ESD, EL, KI, max_iter):
        S_Geral = [TR.copy()]  # Inicializando S_Geral com o conjunto TR
        E = []  # Entropias
        R = []  # Resultado
        iter_num = 1
        sma_values = []  # Valores da SMA
        sd_values = []  # Valores do Desvio Padrão
        num_prototypes = 0  # Contagem de protótipos
        prototypes = []  # Lista para armazenar os protótipos
        sma_values_rep = []  # Valores replicados da SMA
        average_entropies = []  # Média das entropias dos subconjuntos
        accuracy_TR = 0
        accuracy_R = 0

        while iter_num <= max_iter:
            S_Geral_with_classes = [s for s in S_Geral if len(np.unique(s[:, -1])) > 1]
            if not S_Geral_with_classes:
                break

            S_L = max(S_Geral_with_classes, key=lambda subset: subset.shape[0])
            S_Geral = [s for s in S_Geral if not np.array_equal(s, S_L)]

            centroid = self._calculate_centroid(S_L)
            threshold = np.median(euclidean_distances(S_L[:, :-1], centroid.reshape(1, -1)))
            S_1, S_2 = self._split(S_L, centroid, threshold)

            if S_1.size > 0:
                S_Geral.append(S_1)
            if S_2.size > 0:
                S_Geral.append(S_2)

            entropies = [self._calculate_entropy(subset, np.max(TR[:, -1]) + 1) for subset in S_Geral]
            average_entropy = np.mean(entropies)
            average_entropies.append(average_entropy)

            e_i = average_entropy
            E.append(e_i)

            if WMA <= iter_num:
                sma_value = self._sma(E, WMA)
                sma_values.append(sma_value)
                sd_value = self._standard_deviation(E, WMA, sma_value)
                sd_values.append(sd_value)
                sma_values_rep.append(sma_value)
                proto_gen = self._is_proto_generating(WMA, sma_value, E, ESD)
                if proto_gen:
                    break
            iter_num += 1

        for subset in S_Geral:
            if subset.size > 0:
                centroid = self._calculate_centroid(subset)
                entropy_subset = self._calculate_entropy(subset, np.max(TR[:, -1]) + 1)
                if entropy_subset < EL:
                    I = self._is_instance_selecting([subset], centroid, TR, EL, KI)
                    if I.size > 0:
                        num_prototypes += 1
                        R.extend(I)
                        prototypes.append(centroid)
            else:
                print("Subconjunto vazio encontrado, pulando cálculos.")

        R_array = np.array(R)

        accuracy_TR, accuracy_R = self._calculate_accuracy(TR, R_array)
        accuracy_TR *= 100
        accuracy_R *= 100
        reduction_rate = len(R) / len(TR) * 100

        return R, accuracy_TR, accuracy_R, reduction_rate, sma_values, sma_values_rep, average_entropies, sd_values, S_Geral, num_prototypes, prototypes

    # Definindo as funções necessárias
    def _calculate_centroid(self, subset):
        return np.mean(subset[:, :-1], axis=0)

    def _split(self, subset, centroid, threshold):
        distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
        S_1 = subset[distances[:, 0] <= threshold]
        S_2 = subset[distances[:, 0] > threshold]
        return S_1, S_2

    def _select_largest_subset(self, S_1, S_2):
        diameter_1 = np.max(euclidean_distances(S_1[:, :-1], S_1[:, :-1]))
        diameter_2 = np.max(euclidean_distances(S_2[:, :-1], S_2[:, :-1]))
        return S_1 if diameter_1 > diameter_2 else S_2

    def _is_instance_selecting(self,subsets, centroid, training_set, entropy_level, k):
        selected_instances = []
        for subset in subsets:
            if self._calculate_entropy(subset, np.max(training_set[:, -1]) + 1) <= entropy_level:
                distances = euclidean_distances(subset[:, :-1], centroid.reshape(1, -1))
                nearest_indices = np.argsort(distances, axis=0)[:k].flatten()
                selected_instances.extend(subset[nearest_indices])
        return np.array(selected_instances)

    def _calculate_entropy(self, subset, num_classes):
        if len(subset) == 0:
            return 0  # Retorna NaN para subconjuntos vazios
        num_classes = int(num_classes)
        class_counts = np.bincount(subset[:, -1].astype(int), minlength=num_classes)
        probabilities = class_counts / np.sum(class_counts)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        normalized_entropy = entropy / np.log2(num_classes)
        return normalized_entropy if not np.isneginf(normalized_entropy) else 0  # Trata -inf como 0

    def _sma(self, entropy_values, window_size):
        return np.mean(entropy_values[-window_size:])

    def _standard_deviation(self,entropy_values, window_size, sma_value):
        return np.sqrt(np.sum((entropy_values[-window_size:] - sma_value) ** 2) / (window_size - 1))

    def _is_proto_generating(self,window_size, sma_value, entropy_values, esd):
        sd_value = self._standard_deviation(entropy_values, window_size, sma_value)
        #print(sd_value)
        return sd_value < esd

    def _calculate_accuracy(self,TR, R):
        # Dividindo TR em conjunto de treino e validação (90% treino, 10% validação)
        X_TR = TR[:, :-1]  # Características
        y_TR = TR[:, -1]   # Rótulos
        X_train, X_val, y_train, y_val = train_test_split(X_TR, y_TR, test_size=0.1, stratify=y_TR, random_state=42)

        # Treinando o KNN com k=1 com o conjunto de treino de TR e validando com o conjunto de validação
        knn_TR = KNeighborsClassifier(n_neighbors=1)
        knn_TR.fit(X_train, y_train)
        accuracy_TR = knn_TR.score(X_val, y_val)

        # Preparando o conjunto R se não estiver vazio
        accuracy_R = 0
        if len(R) > 0:
            X_R = R[:, :-1]  # Características
            y_R = R[:, -1]   # Rótulos

            # Treinando o KNN com k=1 com o conjunto R e validando com o mesmo conjunto de validação
            knn_R = KNeighborsClassifier(n_neighbors=1)
            knn_R.fit(X_R, y_R)
            accuracy_R = knn_R.score(X_val, y_val)

        return accuracy_TR, accuracy_R

 



