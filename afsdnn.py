import random
def generate_dc_matrix(number_of_attribute, number_of_hidden_neuron):
    dc_matrix = [[0] * number_of_hidden_neuron for _ in range(number_of_attribute)]
    for i in range(number_of_attribute):
        for j in range(number_of_hidden_neuron):
            dc_matrix[i][j] = 0 if random.random() > 0.5 else 1
    return dc_matrix

dc_matrix = generate_dc_matrix(3,2)

print(dc_matrix)

def generate_rule_combination_matrix(number_of_attribute, number_of_fuzzy_set, number_of_hidden_neuron):
    rule_combination_matrix = [[[0] * number_of_hidden_neuron for _ in range(number_of_fuzzy_set)] for _ in range(number_of_attribute)]
    for i in range(number_of_attribute):
        for j in range(number_of_fuzzy_set):
            for k in range(number_of_hidden_neuron):
                rule_combination_matrix[i][j][k] = 0 if random.random() > 0.5 else 1
    return rule_combination_matrix

rule_combination_matrix = generate_rule_combination_matrix(3,3,2)

print(rule_combination_matrix)

def ei_reduce_nd(concepts):
    result = []
    concepts.sort(key=lambda x: len(x))
    for concept in concepts:
        contain = False
        if len(result) == 0:
            result.append(concept)
        else:
            for r in result:
                if set(concept).issuperset(r):
                    contain = True
                    break
            if not contain:
                result.append(concept)
    return result

def ei_multiply_nd(conceptA, conceptB):
    result = []
    for aConcept in conceptA:
        for bConcept in conceptB:
            conceptTmp = set()
            conceptTmp.update(aConcept)
            conceptTmp.update(bConcept)
            result.append(conceptTmp)
    result = ei_reduce_nd(result)
    return result


def ei_sum_nd(concept_a, concept_b):
    result = []
    result.extend(concept_a)
    result.extend(concept_b)
    result = ei_reduce_nd(result)
    return result


conceptA = [
    {1, 2, 3},
    {3, 4},
    {5, 6},
    {1, 2, 3},
    {7, 8, 9}
]

conceptB = [
    {1, 2, 3},
    {3, 5},
    {5, 7},
    {1, 2, 3},
    {7, 8, 9}
]

ans = ei_multiply_nd(conceptA,conceptB)
print(ans)


# * @ param concept 逻辑与的概念集合，我们用set来表示简单概念之间的逻辑与
# * @ param sampleIndex 样本索引
# * @ param structure afs结构对象
# * @ return degree 关系指标
# * @ description
# 该方法用于计算样本在逻辑与的简单概念set集合上的隶属度（membership
# degree）


# 1. `mat = set()`：创建一个空的集合，用于存储中间结果。
#
# 2. `tau = bytearray(len(structure.getData()) // 8 + 1)`：
# 根据`structure`中的数据长度计算出所需的`bytearray`的长度，并初始化为相应长度的`tau`。
#
# 3. `for i in range(len(tau)): tau[i] = -1`：将`tau`中的每个元素初始化为-1。
#
# 4. `degree = 1`：初始化隶属度为1。
#
# 5. `for simpleConcept in concept:`：对于每个概念中的简单概念，执行以下操作：
#
#    a. `tmpKey = [sampleIndex, simpleConcept.conceptIndex]`：创建一个临时键值对，用于获取`structure`中的`tau`。
#
#    b. `tmpTau = structure.getTau()[simpleConcept.conceptIndex][sampleIndex]`：从`structure`中获取指定概念和样本索引的`tau`。
#
#    c. `for i in range(len(tau)): tau[i] = tau[i] & tmpTau[i]`：将每个元素与对应`tmpTau`中的元素进行按位与操作，并更新`tau`中的值。
#
# 6. `for simpleConcept in concept:`：对于每个概念中的简单概念，执行以下操作：
#
#    a. `weightMat = structure.weight[simpleConcept.conceptIndex]`：获取`structure`中指定概念的权重矩阵。
#
#    b. `weightBoolSum = 0`：初始化布尔权重和为0。
#
#    c. `for i in range(len(structure.getData())):`：遍历`structure`中数据的每个元素，执行以下操作：
#
#       - `if (tau[i//8] & (1 << (i%8))) != 0:`：检查`tau`中的对应位是否为1。
#
#          - 如果为1，则执行以下操作：
#
#            - `weightBoolSum += weightMat[i]`：将对应位置上的权重值加到布尔权重和中。
#
#    d. `weightSum = structure.weightSum[simpleConcept.conceptIndex] + 0.00000000001`：获取指定概念的权重和，并加上一个极小值。
#
#    e. `sumDegree = weightBoolSum / weightSum`：计算简单概念的隶属度。
#
#    f. `degree *= sumDegree`：将简单概念的隶属度乘以总隶属度。
#
# 7. `return degree`：返回计算得到的总隶属度。

def getMembershipDegreeWedge_nD(concept, sampleIndex, structure):
    mat = set()
    tau = bytearray((len(structure) + 7) // 8)
    for i in range(len(tau)):
        tau[i] = 255

    degree = 1
    for simpleConcept in concept:
        tmpKey = [sampleIndex, simpleConcept['conceptIndex']]
        tmpTau = structure[str(simpleConcept['conceptIndex'])][sampleIndex]
        for i in range(len(tau)):
            tau[i] = tau[i] & tmpTau[i]
    for simpleConcept in concept:
        weightMat = simpleConcept['weightMat']
        weightBoolSum = 0
        for i in range(len(structure)):
            if (tau[i // 8] & (1 << (i % 8))) != 0:
                weightBoolSum += weightMat[i]
        weightSum = structure['weightSum'][simpleConcept['conceptIndex']] + 0.00000000001
        sumDegree = weightBoolSum / weightSum
        degree *= sumDegree
    return degree
# 示例的输入
structure = {
    '0': {
        'sampleIndex': [1, 0, 1, 0, 1],
        'weightSum': 10
    },
    '1': {
        'sampleIndex': [0, 1, 0, 1, 0],
        'weightSum': 15
    },
    '2': {
        'sampleIndex': [1, 1, 1, 1, 1],
        'weightSum': 8
    }
}

concept = [
    {
        'conceptIndex': '0',
        'weightMat': [0, 1, 0, 1, 0]
    },
    {
        'conceptIndex': '1',
        'weightMat': [1, 0, 1, 0, 1]
    }
]

sampleIndex = 2


# * @ param hiddenLayerResults 隐层结果
# * @ param hiddenLayerDescriptions 隐层描述
# * @ return 类描述
# * @ description 通过结果获取类描述


#
# 接受两个参数hiddenLayerResults和hiddenLayerDescriptions，并返回一个Map>>类型的结果classDescription。
#
# 在这个方法中，首先创建了一个空的classDescription，用于存储类别描述。然后通过遍历hiddenLayerResults中的每个类别，进行以下操作：
#
# 1. 获取当前类别内的样本编号列表inClass。
# 2. 创建一个空的样本编号列表outOfClass，用于存储其他类别的样本编号。
# 3. 遍历所有类别，将不是当前类别的样本编号添加到outOfClass列表中。
# 4. 针对当前类别内的每个样本，计算类内均值和类外均值，并找出评价指标D_c_i最大值的样本编号maxIndexD和评价指标F_c_i最大值的样本编号maxIndexF。
# 5. 使用maxIndexD和maxIndexF对应的样本，通过调用ei.eiSumNd方法，计算出一个描述and。
# 6. 将当前类别编号cn和描述and添加到classDescription中。
# 7. 循环结束后，返回classDescription作为方法的结果。
#
# 总体来说，这段代码的目的是根据给定的样本编号和样本描述，计算出每个类别的描述，并将结果存储在classDescription中返回。具体的计算过程涉及到对类内均值、类外均值和评价指标的比较，以及调用ei.eiSumNd方法进行描述的生成。

def get_membership_degree_nD(concept, sampleIndex, structure):
    weightTmp = []
    for andSimpleConcept in concept:
        singleDegree = getMembershipDegreeWedge_nD(andSimpleConcept, sampleIndex, structure)
        weightTmp.append(singleDegree)
    orMembershipDegree = max(weightTmp) if weightTmp else 0.0
    return orMembershipDegree

def get_description_for_class(hidden_layer_results, hidden_layer_descriptions):
    class_description = {}
    class_numbers = hidden_layer_results.keys()

    for cn in class_numbers:
        # 类内
        in_class = hidden_layer_results[cn]

        # 类外
        out_of_class = []
        for cnn in class_numbers:
            if cnn != cn:
                out_of_class.extend(hidden_layer_results[cnn])

        # 从中找到评价指标D_c_i最大值的描述
        max_index_d = 0
        max_average = float('-inf')
        max_index_f = 0
        max_degree = float('-inf')

        for s in in_class:
            # 类内均值
            in_average = 0
            for ss in in_class:
                in_average += get_membership_degree_nD(hidden_layer_descriptions[s], ss, structure)
            in_average /= len(in_class)

            # 类外均值
            out_average = 0
            for sss in out_of_class:
                out_average += get_membership_degree_nD(hidden_layer_descriptions[s], sss, structure)
            out_average /= len(out_of_class)

            # max_index_d中存储的时指标D_c_i最大对应的样本标号
            if (in_average - out_average) > max_average:
                max_average = in_average - out_average
                max_index_d = s

            # 从样本中找到评价指标F_c_i最大的样本

            # 类内最大
            in_degree = float('-inf')
            for tt in in_class:
                tt_degree = get_membership_degree_nD(hidden_layer_descriptions[s], tt, structure)
                if tt_degree > in_degree:
                    in_degree = tt_degree

            # 类外最大
            out_degree = float('-inf')
            for ttt in out_of_class:
                ttt_degree = get_membership_degree_nD(hidden_layer_descriptions[s], ttt, structure)
                if ttt_degree > out_degree:
                    out_degree = ttt_degree

            # max_index_f中存储的时指标F_c_i最大对应的样本标号
            if (in_degree - out_degree) > max_degree:
                max_degree = in_degree - out_degree
                max_index_f = s

        # 将D_c_i和F_c_i值最大对应的两个样本做EI代数和形成c_i类的描述
        and_ = ei_sum_nd(hidden_layer_descriptions[max_index_d], hidden_layer_descriptions[max_index_f])

        class_description[cn] = and_

    return class_description





def generate_complex_concepts(instances, DCMatrix, ruleCombinationsMatrix, outputWeight,
                              nFeatures,
                              numberOfFuzzySet,
                              simpleConceptObjects):
    hiddenLayerDescriptions = []
    hiddenLayerResults = {}
    for i in range(len(outputWeight)):
        flag = 0
        for p in range(len(outputWeight[i])):
            if outputWeight[i][p] > outputWeight[i][flag]:
                flag = p

        if flag not in hiddenLayerResults:
            hiddenLayerResults[flag] = []
        hiddenLayerResults[flag].append(i)
        # 其中 i 值是最大值的索引
        # 将健值序列i 输入到 flag 序列中去

        tl = []
        for j in range(nFeatures):
            if DCMatrix[j][i] == 1:
                continue
            else:
                tf = []
                for k in range(numberOfFuzzySet):
                    if ruleCombinationsMatrix[j][k][i] == 1:
                        t = set()
                        t.add(simpleConceptObjects[j * numberOfFuzzySet + k])
                        tf.append(t)
                if len(tl) == 0:
                    tl.extend(tf)
                else:
                    tl = ei_multiply_nd(tl, tf)
        hiddenLayerDescriptions.append(tl)

    map = get_description_for_class(hiddenLayerResults, hiddenLayerDescriptions)

    return map

import  numpy as np

import numpy as np


def getDataDMatrix(instances):
    numericAttNum = 0
    for attNum in range(instances.numAttributes()):
        attribute = instances.attribute(attNum)
        if attribute.isNumeric():
            numericAttNum += 1
        else:
            instances.setClassIndex(attNum)

    sampleNum = instances.size()
    data = np.zeros((sampleNum, numericAttNum))

    for i in range(sampleNum):
        cur = 0
        for j in range(numericAttNum):
            if cur != instances.classIndex():
                data[i][j] = instances.instance(i).value(cur)
                j += 1
            cur += 1

    return data

def getClassValues(instances):
    result = []
    if instances.classIndex() < 0:
        for attNum in range(instances.numAttributes()):
            attribute = instances.attribute(attNum)
            if attribute.isNominal() and attribute.name() == "class":
                instances.setClassIndex(attNum)
        for i in range(instances.numInstances()):
            classValueD = instances.instance(i).classValue()
            classValueI = int(classValueD)
            result.append(classValueI)
            # print(classValueI)
    else:
        for i in range(instances.numInstances()):
            classValueD = instances.instance(i).classValue()
            classValueI = int(classValueD)
            result.append(classValueI)
            # print(classValueI)
    return result





def getTargetDMatrix(instances):
    classValues = getClassValues(instances)
    if classValues is None:
        return None
    classNum = max(classValues) + 1
    target = [[0.0] * classNum for _ in range(len(classValues))]
    for i in range(len(classValues)):
        for j in range(classNum):
            if classValues[i] == j:
                target[i][j] = 1.0
            else:
                target[i][j] = 0.0
    return target


def build_classifier(instances):
    i, j, k, l = 0, 0, 0, 0
    train_instances = instances
    data = getDataDMatrix(instances)
    target = getDataDMatrix(instances)
    DCMatrix = generate_DCMatrix(nFeatures, numberOfHiddenNeuron)
    ruleCombinationMatrix = generate_rule_combination_matrix(nFeatures, numberOfFuzzySet, numberOfHiddenNeuron)

    # 初始化AFS结构类
    afs = CutPointStructure_nD(data, target, simpleConceptMap, weightMap)
    # 构建AFS结构
    afs.generate_structure()

    ei = EIAlgebra()
    mf = CutPointMembershipFunction()
    init_classifier(ei, mf, afs)

    PROBORValue = np.zeros((instances.numInstances(), nFeatures, numberOfHiddenNeuron))
    for l in range(instances.numInstances()):
        for i in range(nFeatures):
            for j in range(numberOfHiddenNeuron):
                if DCMatrix[i][j] == 1:
                    PROBORValue[l][i][j] = 1.0
                else:
                    ithAttributeSimpleConcepts = simpleConceptObjects[i*numberOfFuzzySet:(i+1)*numberOfFuzzySet]
                    product = 1
                    for k in range(numberOfFuzzySet):
                        simpleConcept = ithAttributeSimpleConcepts[k]
                        simpleConceptSet = set()
                        simpleConceptSet.add(simpleConcept)
                        description = []
                        description.append(simpleConceptSet)
                        product *= (1 - ruleCombinationMatrix[i][k][j] * mf.get_membership_degree_nD(description, l, structure))
                    PROBORValue[l][i][j] = 1 - product

    firingStrength = np.zeros((instances.numInstances(), numberOfHiddenNeuron))
    for i in range(instances.numInstances()):
        for j in range(numberOfHiddenNeuron):
            product = 1
            for k in range(nFeatures):
                product *= PROBORValue[i][k][j]
            firingStrength[i][j] = product

    H = DenseMatrix(firingStrength)
    invers = Inverse(H)
    pinvHt = invers.get_MP_inverse(0.000001)
    T = DenseMatrix(instances.numInstances(), numberOfOutputNeuron)
    for i in range(instances.numInstances()):
        for j in range(numberOfOutputNeuron):
            if j == int(instances.instance(i).classValue()):
                T.set(i, j, 1.0)
            else:
                T.set(i, j, -1.0)

    OutputWeight = DenseMatrix(numberOfHiddenNeuron, numberOfOutputNeuron)
    pinvHt.mult(T, OutputWeight)

    outputWeight = OutputWeight
    DCMatrix = DCMatrix
    ruleCombinationMatrix = ruleCombinationMatrix
    description = generate_complex_concepts(instances, DCMatrix, ruleCombinationMatrix, outputWeight)

    keys = description.keys()
    for key in keys:
        print("第" + str(key) + "类描述")
        arr = ei.ei_reduce_nd(description[key])
        andFlag = True
        for ar in arr:
            if andFlag:
                andFlag = False
            else:
                print("+", end="")
            orFlag = True
            for kk in ar:
                if orFlag:
                    orFlag = False
                    print(kk, end="")
                else:
                    print("*" + kk, end="")
        print("\n=================================")
