from random import seed
from random import randrange


# Calcular el porcentaje de precisión
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Dividir un conjunto de datos basado en un atributo y un valor de atributo
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calcular el índice de Gini para un conjunto de datos dividido
def gini_index(groups, classes):
    # Contar todas las muestras en el punto de división
    n_instances = float(sum([len(group) for group in groups]))
    # Sumar el índice de Gini ponderado para cada grupo
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # Evitar división por cero
        if size == 0:
            continue
        score = 0.0
        # Calcular la puntuación del grupo basado en la puntuación para cada clase
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # Ponderar la puntuación del grupo por su tamaño relativo
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Seleccionar el mejor punto de división para un conjunto de datos
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


# Crear un valor de nodo terminal
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Crear divisiones de hijos para un nodo o hacer terminal
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del node["groups"]
    # Verificar si no hay división
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    # Verificar la profundidad máxima
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    # Procesar hijo izquierdo
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth + 1)
    # Procesar hijo derecho
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


# Construir un árbol de decisión
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Hacer una predicción con un árbol de decisión
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# Algoritmo de Árbol de Clasificación y Regresión
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions, tree
