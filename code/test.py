from sklearn.metrics.pairwise import cosine_similarity

# from scipy.stats import entropy

x = [0, 1, 1]
y = [1, 0]
z = [1, 0]
print(cosine_similarity(x, y) + cosine_similarity(z, [1, 1]))
# # print(entropy(dataSetI,dataSetII))
#
# # cwd = os.getcwd()
# # print(cwd)
# # for file in os.listdir('../'):
# #     print(file)
#
# self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
#     probability_fn(
#         _maybe_mask_score(score, memory_sequence_length, score_mask_value),
#         prev))

a = (1, 2, 4)
if len(a) > 2:
    print(a[2])

a = 5, 7, 5, 6
len(a)
