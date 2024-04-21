import numpy as np
from sklearn.datasets import load_iris
import random
class EIAlgebra():
    print('')

class CutPointMembershipFunction():
    print('')

def getMembershipDegreeWedge_nD():
    print('')




def CutPointStructure_nD():
    print('')



def generate_DC_matrix(number_of_attribute, number_of_hidden_neuron):
    dc_matrix = [[0] * number_of_hidden_neuron for _ in range(number_of_attribute)]
    for i in range(number_of_attribute):
        for j in range(number_of_hidden_neuron):
            dc_matrix[i][j] = 0 if random.random() > 0.5 else 1
    return dc_matrix




def generate_rule_combination_matrix(number_of_attribute, number_of_fuzzy_set, number_of_hidden_neuron):
    rule_combination_matrix = [[[0] * number_of_hidden_neuron for _ in range(number_of_fuzzy_set)] for _ in range(number_of_attribute)]
    for i in range(number_of_attribute):
        for j in range(number_of_fuzzy_set):
            for k in range(number_of_hidden_neuron):
                rule_combination_matrix[i][j][k] = 0 if random.random() > 0.5 else 1
    return rule_combination_matrix



def getDataDMatrix(data):
    numericAttNum = data.shape[1]  # 获取数值属性的数量
    sampleNum = data.shape[0]  # 获取样本数量
    data_matrix = np.zeros((sampleNum, numericAttNum))

    for i in range(sampleNum):
        data_matrix[i] = data[i, :numericAttNum]

    return data_matrix

# 加载 Iris 数据集
iris = load_iris()
data = iris.data

# 使用您的函数来获取数值属性数据
data_matrix = getDataDMatrix(data)

# 输出结果
# print(data_matrix)




from sklearn.datasets import load_iris

def getClassValues(target):
    result = []

    for class_value in target:
        result.append(class_value)

    return result

# 加载 Iris 数据集
iris = load_iris()
target = iris.target

# 使用您的函数来获取类别值
class_values = getClassValues(target)

# 输出结果


def getTargetDMatrix(target):
    classValues = getClassValues(target)
    if classValues is None:
        return None
    classNum = max(classValues) + 1
    target_matrix = [[0.0] * classNum for _ in range(len(classValues))]
    for i in range(len(classValues)):
        for j in range(classNum):
            if classValues[i] == j:
                target_matrix[i][j] = 1.0
            else:
                target_matrix[i][j] = 0.0
    return target_matrix

# 加载 Iris 数据集
iris = load_iris()
target = iris.target

# 使用您的函数来获取目标矩阵
target_matrix = getTargetDMatrix(target)

# 输出结果
# for row in target_matrix:
#     print(row)


def process_instances(target):
    classValues = getClassValues(target)
    if classValues is None:
        return None
    classNum = max(classValues) + 1
    target_matrix = [[0.0] * classNum for _ in range(len(classValues))]
    for i in range(len(classValues)):
        for j in range(classNum):
            if classValues[i] == j:
                target_matrix[i][j] = 1.0
            else:
                target_matrix[i][j] = 0.0
    return target_matrix

# 加载 Iris 数据集
iris = load_iris()
target = iris.target

# 使用您的函数来处理实例数据
processed_target = process_instances(target)

# 输出结果
for row in processed_target:
    print(row)


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


def get_membership_degree_nD(concept, sampleIndex, structure):
    weightTmp = []
    for andSimpleConcept in concept:
        singleDegree = getMembershipDegreeWedge_nD(andSimpleConcept, sampleIndex, structure)
        weightTmp.append(singleDegree)
    orMembershipDegree = max(weightTmp) if weightTmp else 0.0
    return orMembershipDegree

def get_description_for_class(hidden_layer_results, hidden_layer_descriptions,structure):
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


'''
number of output neuron = 分类器种类的数量

instances = load_iris
'''

# 
# # 示例的输入
# structure = {
#     '0': {
#         'sampleIndex': [1, 0, 1, 0, 1],
#         'weightSum': 10
#     },
#     '1': {
#         'sampleIndex': [0, 1, 0, 1, 0],
#         'weightSum': 15
#     },
#     '2': {
#         'sampleIndex': [1, 1, 1, 1, 1],
#         'weightSum': 8
#     }
# }
def buildClassifier(instances, numClasses,numberOfHiddenNeuron, numberOfOutputNeuron, 
                    sigma, nFeatures, structure, outputWeight):
    dataa = instances.data

    targeta = instances.target

    data = getDataDMatrix(dataa)

    target = getTargetDMatrix(targeta)

    afs = CutPointStructure_nD(data, target)

    afs.generateConceptsAndWeights(sigma)

    simpleConceptMap = afs.concepts

    numberOfFuzzySet = len(simpleConceptMap) // nFeatures

    print("numberOfFuzzySet:", numberOfFuzzySet)

    simpleConceptObjects = sorted(list(simpleConceptMap.values()), key=lambda o: o.conceptIndex)

    weightMap = afs.weightFunction

    afs.generateStructure()

    DCMatrix = generate_DC_matrix(nFeatures, numberOfHiddenNeuron)

    ruleCombinationMatrix = generate_rule_combination_matrix(nFeatures, numberOfFuzzySet,
                                                               numberOfHiddenNeuron)

    # Initialize EIAlgebra and CutPointMembershipFunction
    ei = EIAlgebra()
    mf = CutPointMembershipFunction()
    # initClassifier(ei, mf, afs)

    num_instances = dataa.shape[0]
    
    
    # Create a 3D array PROBORValue with dimensions (numInstances, nFeatures, numberOfHiddenNeuron)
    import numpy as np
    
    PROBORValue = np.zeros((num_instances, nFeatures, numberOfHiddenNeuron))
    for l in range(num_instances):
        for i in range(nFeatures):
            for j in range(numberOfHiddenNeuron):
                if DCMatrix[i][j] == 1:
                    PROBORValue[l][i][j] = 1.0
                else:
                    ithAttributeSimpleConcepts = simpleConceptObjects[
                                                 i * numberOfFuzzySet: (i + 1) * numberOfFuzzySet]

                    ## 第i个特征  我们可以从 simpleConceptObjects 中进行切片。 包含 numberOfFuzzySet 个元素。

                    product = 1.0

                    for k in range(numberOfFuzzySet):
                        simpleConcept = ithAttributeSimpleConcepts[k]
                        simpleConceptSet = {simpleConcept}
                        description = [simpleConceptSet]
                        product *= (1 - ruleCombinationMatrix[i][k][j] * mf.getMembershipDegree_nD(description, l,
                                                                                                        structure))
                    PROBORValue[l][i][j] = 1 - product

            # 复杂概念是否被忽略 同时构建三维 P 数组

            import numpy as np
            from numpy.linalg import inv, pinv

            # Create a 2D array firingStrength with dimensions (numInstances, numberOfHiddenNeuron)

            firingStrength = np.zeros((instances.numInstances(), numberOfHiddenNeuron))

            for i in range(instances.numInstances()):
                for j in range(numberOfHiddenNeuron):
                    product = 1.0
                    for k in range(nFeatures):
                        product *= PROBORValue[i][k][j]
                    firingStrength[i][j] = product
            # 计算神经元是否激活

            # Create a dense matrix H using firingStrength
            H = np.matrix(firingStrength)

            # Calculate the Moore-Penrose pseudoinverse of H
            pinvHt = pinv(H + np.identity(H.shape[1]) * 0.000001)

            # Create a dense matrix T with dimensions (numInstances, numberOfOutputNeuron)
            T = np.zeros((instances.numInstances(), numberOfOutputNeuron))

            for i in range(instances.numInstances()):
                for j in range(numberOfOutputNeuron):
                    if j == int(instances.instance(i).classValue()):
                        T[i][j] = 1.0
                    else:
                        T[i][j] = -1.0

            from collections import defaultdict

            # Multiply pinvHt and T to get the OutputWeight
            OutputWeight = np.dot(pinvHt, T)

            # 输出权重是 pinvht 和 T 的点乘

            description = generate_complex_concepts(instances, DCMatrix, ruleCombinationMatrix, outputWeight)

            keys = description.keys()
            for key in keys:
                print("第" + str(key) + "类描述")
                arr = ei.eiReduceNd(description[key])
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
                
            def classify_instance(self, instance):
                PROBORValue = np.zeros((self.nFeatures, self.numberOfHiddenNeuron))

                instances = self.trainInstances + [instance]

                data = getDataDMatrix(instances)
                target = getTargetDMatrix(instances)

                afs_test = CutPointStructure_nD(data, target)
                afs_test.generateConceptsAndWeights(self.sigma)
                afs_test.generateStructure()

                for i in range(self.nFeatures):
                    for j in range(self.numberOfHiddenNeuron):
                        if self.DCMatrix[i][j] == 1:
                            PROBORValue[i][j] = 1.0
                        else:
                            ith_attribute_simple_concepts = self.simpleConceptObjects[
                                                            i * self.numberOfFuzzySet: (i + 1) * self.numberOfFuzzySet]
                            product = 1.0
                            for k in range(self.numberOfFuzzySet):
                                simpleConcept = ith_attribute_simple_concepts[k]
                                simpleConceptSet = {simpleConcept}
                                description = [simpleConceptSet]
                                product *= (1 - self.ruleCombinationMatrix[i][k][j] * self.mf.getMembershipDegree_nD(
                                    description, len(instances) - 1, afs_test))
                            PROBORValue[i][j] = 1 - product


                firingStrength = np.zeros(self.numberOfHiddenNeuron)

                for i in range(self.numberOfHiddenNeuron):
                    product = 1.0
                    for j in range(self.nFeatures):
                        product *= PROBORValue[j][i]






                    firingStrength[i] = product

                output = np.zeros(self.numberOfOutputNeuron)

                for i in range(self.numberOfOutputNeuron):
                    sum_output = 0.0
                    for j in range(self.numberOfHiddenNeuron):
                        sum_output += self.outputWeight[j, i] * firingStrength[j]
                    output[i] = sum_output

                flag = 0

                for i in range(self.numberOfOutputNeuron):
                    if output[i] > output[flag]:
                        flag = i

                classIndex = flag
                return classIndex

            # 接下来，通过循环找到 output 数组中具有最大值的元素索引，将其赋值给 flag 变量。这就确定了神经网络输出中具有最大值的输出神经元，
            # 从而确定了最终的分类索引。

            def class_description(self):
                class_description = self.description

                result = []
                keys = class_description.keys()
                for k in keys:
                    arr = class_description[k]
                    result.append(f"Class Description for Class {k + 1}: ")
                    and_flag = True
                    for ar in arr:
                        if and_flag:
                            and_flag = False
                        else:
                            result.append("+")
                        or_flag = True
                        for kk in ar:
                            if or_flag:
                                or_flag = False
                                result.append(str(kk.concept_index))
                            else:
                                result.append("*" + str(kk.concept_index))
                    result.append("\n")
                return ''.join(result)












