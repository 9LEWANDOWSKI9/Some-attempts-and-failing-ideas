import afsdu.memberFunctions
import afsdu.simpleConcepts
import afsdu.structures
import afsdu.utils.UtilsDu
from weka.core import Instances
import decimal
from java.util import ArrayList

class CutPointMembershipFunction(MembershipFunction):
    def get_membership_degree_wedge_nD(self, concept, sample_index, structure):
        mat = set()  # 声明一个mat集，元素类型是string，用set来存储集合
        tau = bytearray(structure.getData().length // 8 + 1)  # 声明了tau数组，类型为bytearray型
        tau[:] = bytes([255])  # 用0xFF填充整个数组
        degree = 1.0
        for simpleConcept in concept:
            tmp_key = [sample_index, simpleConcept.conceptIndex]
            tmp_tau = structure.get_tau()[simpleConcept.conceptIndex][sample_index]
            for i in range(len(tau)):
                tau[i] &= tmp_tau[i]

        for simpleConcept in concept:
            weight_mat = structure.weight[simpleConcept.conceptIndex]
            weight_bool_sum = 0.0
            for i in range(len(structure.getData())):
                if (tau[i // 8] & (1 << (i % 8))) != 0:
                    weight_bool_sum += weight_mat[i]
            weight_sum = structure.weight_sum[simpleConcept.conceptIndex] + 0.00000000001
            sum_degree = Decimal(weight_bool_sum) / Decimal(weight_sum)
            degree *= sum_degree

        return degree


def get_membership_degree_of_simple_concept(self, simpleConcept, structure):
    degree = []
    weightMat = structure.weight[simpleConcept.conceptIndex]
    for sampleIndex in range(len(structure.getData())):
        tau = structure.get_tau()[simpleConcept.conceptIndex][sampleIndex]
        weightBoolSum = sum(weightMat[x] for x in range(len(structure.getData())) if (tau[x // 8] & (1 << (x % 8))) == tau[x // 8])
        weightSum = structure.weight_sum[simpleConcept.conceptIndex]
        d1 = Decimal(weightBoolSum)
        d2 = Decimal(weightSum)
        degree.append(d1 / d2)
    return degree

def get_membership_degree_nD(self, concept, sample_index, structure):
    weight_tmp = []
    for and_simple_concept in concept:
        single_degree = self.get_membership_degree_wedge_nD(and_simple_concept, sample_index, structure)
        weight_tmp.append(single_degree)
    or_membership_degree = max(weight_tmp) if weight_tmp else 0.0
    return or_membership_degree


def get_membership_degrees_nD(self, concepts, structure):
    degrees = []
    for i in range(len(structure.getData())):
        degrees.append(self.get_membership_degree_nD(concepts, i, structure))
    return degrees

def main():
    a = UtilsDu.get_instances_for_arff_file("Data/iris.arff")
    data = UtilsDu.get_data_d_matrix(a)
    target = UtilsDu.get_target_d_matrix(a)
    cut_point_concepts = CutPointConcepts_nD()
    concept_string = "13 2 1"
    cut_point_concepts.set_simple_concepts_nD(concept_string)
    # afs = CutPointStructure_nD(data, target, cut_point_concepts.get_simple_concepts_nD())
    # afs.generate_structure()
    # mf = CutPointMembershipFunction()
    # set = set(cut_point_concepts.get_all_simple_concepts())
    # print(mf.get_membership_degree_wedge(set, 1, afs))
    # b1 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 2) | (1 << 1) | (1 << 0)
    # print(b1)

if __name__ == "__main__":
    main()
