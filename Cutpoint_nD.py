
import numpy as np


class CutPointStructure_nD(AbstractStructure, Serializable):
    def __init__(self, data, target=None, concepts=None, weightFunction=None):
        super().__init__(data, target)
        self.tau = {}  # tau is initialized as a dictionary

        self.concepts = concepts if concepts is not None else {}
        self.weightFunction = weightFunction if weightFunction is not None else {}
        self.weight = {}  # calculated by weightFunction
        self.weightSum = {}

def generate_concepts_and_weights(self, sigma):
    self.concepts = {}
    self.weightFunction = {}
    n_feature = len(self.data[0])
    n_instance = len(self.data)

    concept_index = 1
    for j in range(n_feature):
        _sum = 0
        _min = float('inf')
        _max = float('-inf')
        for i in range(n_instance):
            value = self.data[i][j]
            _min = min(_min, value)
            _max = max(_max, value)
            _sum += value
        mean = _sum / n_instance
        concept_index = self.add_concept_and_weight(sigma, concept_index, j, _min)
        concept_index = self.add_concept_and_weight(sigma, concept_index, j, mean)
        concept_index = self.add_concept_and_weight(sigma, concept_index, j, _max)


def add_concept_and_weight(self, sigma, concept_index, j, value):
    simple_concept = CutPointConcept_nD(concept_index, [j], [value])
    self.concepts[concept_index] = simple_concept
    weight_function_str = f"{concept_index} 1 {j} {value}:gaussmf 1 {sigma}"
    self.weightFunction[concept_index] = weight_function_str
    concept_index += 2
    return concept_index


def generate_structure(self):
    samples = self.data
    new_concepts = list(self.concepts.keys())

    for simple_concept_index in new_concepts:
        weight_val = [0.0] * len(samples)
        tmp_map = {}
        self.tau[simple_concept_index] = tmp_map

        for i, i_val in enumerate(samples):
            sample_indexes = bytearray((len(samples) + 7) // 8)
            current_simple_concept = self.concepts[simple_concept_index]
            if simple_concept_index % 2 == 0:
                for j, j_val in enumerate(samples):
                    distance1 = 0
                    distance2 = 0
                    for k in range(len(current_simple_concept.feature_index)):
                        distance1 += (i_val[current_simple_concept.feature_index[k]] -
                                      current_simple_concept.feature_description[k]) ** 2
                        distance2 += (j_val[current_simple_concept.feature_index[k]] -
                                      current_simple_concept.feature_description[k]) ** 2
                    if distance1 > distance2:
                        sample_indexes[j // 8] |= 1 << (j % 8)
                    tmp_map[i] = sample_indexes[:]

                current_weight_function = self.weightFunction[simple_concept_index]
                feature = [i_val[k] for k in current_simple_concept.feature_index]
                weight_tmp = self.get_weight_nd(current_weight_function, feature)
                weight_val[i] = weight_tmp
            else:
                for j, j_val in enumerate(samples):
                    distance1 = 0
                    distance2 = 0
                    for k in range(len(current_simple_concept.feature_index)):
                        distance1 += (i_val[current_simple_concept.feature_index[k]] -
                                      current_simple_concept.feature_description[k]) ** 2
                        distance2 += (j_val[current_simple_concept.feature_index[k]] -
                                      current_simple_concept.feature_description[k]) ** 2
                    if distance1 <= distance2:
                        sample_indexes[j // 8] |= 1 << (j % 8)
                    tmp_map[i] = sample_indexes[:]

                current_weight_function = self.weightFunction[simple_concept_index]
                feature = [i_val[k] for k in current_simple_concept.feature_index]
                weight_tmp = self.get_weight_nd(current_weight_function, feature)
                weight_val[i] = weight_tmp

        self.weight[simple_concept_index] = weight_val
        weight_sum = sum(weight_val)
        self.weightSum[simple_concept_index] = weight_sum


def get_weight(self, current_weight_function, feature):
    para_list = current_weight_function.split(" ")
    nD = int(para_list[1])
    result = 0.0
    u = [float(para_list[nD + 2 + i]) for i in range(nD)]
    weight_method = para_list[2 + 2 * nD]
    Q = [float(para_list[3 + 2 * nD + i]) for i in range(nD * nD)]
    parameters = [u, Q]
    result = eval_fuzzy_set(feature, weight_method, parameters)
    return result


def get_weight_nd(current_weight_function, feature):
    # Split the sample value and weight formula
    split = current_weight_function.split(":")
    sample_value_string = split[0].split(" ")
    weight_value_strings = split[1].split(";")

    nD = int(sample_value_string[1])
    weight_strings = [w.split(" ") for w in weight_value_strings]
    weights = []
    count = 2 + nD
    feature_count = 0

    for strings in weight_strings:
        number = int(strings[1])
        parameters = []

        strings1 = strings[2].split(",")
        para = [float(p) for p in strings1]

        if strings[0] == "gaussmf":
            cut_points = [float(sample_value_string[j]) for j in range(count, count + number)]
            parameters.append(cut_points)
        parameters.append(para)
        count += number
        weight = eval_fuzzy_set(feature[feature_count:feature_count + number], strings[0], parameters)
        feature_count += number
        weights.append(weight)

    weight_product = 1.0
    for w in weights:
        weight_product *= w

    return weight_product

def main():
    str = "13 3 1 2 3 1.5 2.5 3.1:gaussmf 2 1.0,0.0,0.0,1.0;trimf 1 4.0,5.0,6.0,0.0"
    weight_nd = get_weight_nd(str, [1.0, 2.0, 4.5])
    print("WeightNd:", weight_nd)

if __name__ == "__main__":
    main()




data = np.array([1, 2, 3])
# ints = data[0:2]  # Equivalent to Java's Arrays.copyOfRange
# print(ints)

# simpleconcepts = PersistenceUtil.readTxt("data/simpleconcept_nD.txt")
# print(simpleconcepts)
# weights = PersistenceUtil.readTxt("data/weight_nD.txt")
# print(weights)
# concepts = Transformer.transform_cut_point_input_nD(simpleconcepts)
# parse_weights = Transformer.transform_weights(weights)
# print(parse_weights)
# filepath = "data/iris.arff"
# instances = UtilsDu.get_instances_for_arff_file(filepath)
# data = UtilsDu.instances_to_list(instances)
# target = UtilsDu.get_target(instances)
# afs = CutPointStructure_nD(data, target, concepts, parse_weights)
# afs.generate_structure()
# print(afs.weight)



