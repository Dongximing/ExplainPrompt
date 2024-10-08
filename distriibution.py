
import matplotlib.pyplot as plt

# Example list of data
data = [0.956043956043956, 0.8642857142857141, 0.9472527472527472, 0.9736263736263736, 0.9720279720279721, 0.8531468531468532, 0.9020979020979022, 0.9780219780219779, 0.8461538461538463, 0.8186274509803922, 0.8499999999999999, 0.9340659340659341, 0.9725274725274725, 0.976470588235294, 0.8647058823529412, 0.8272727272727273, 0.8901098901098902, 0.8571428571428571, 0.8321678321678322, 0.9912087912087912, 0.8472652218782251, 0.8956043956043955, 0.8186813186813187, 0.7749999999999999, 0.8131868131868131, 0.9580419580419581, 0.9757575757575757, 0.8626373626373626, 0.8681318681318682, 0.8846153846153846, 1.0, 0.9930069930069931, 0.7902097902097903, 0.9120879120879121, 0.9208791208791209, 0.961764705882353, 0.8736842105263157, 0.8754385964912281, 0.9835164835164836, 0.9230769230769231, 0.9160839160839163, 0.9363636363636365, 0.9164835164835166, 0.7701754385964912, 0.8406593406593407, 0.8956043956043955, 0.7714285714285714, 0.9440559440559443, 0.989010989010989, 0.9615384615384615, 0.9285714285714284, 0.9692307692307693, 0.8461538461538463, 1.0, 0.7342657342657343, 0.8791208791208791, 0.8791208791208791, 0.901098901098901, 0.8741258741258742, 0.9510489510489512, 0.6648351648351648, 0.8464285714285712, 0.9648351648351648, 0.9945054945054945, 0.9508771929824561, 0.9340659340659341, 0.945054945054945, 0.9790209790209792, 0.9107142857142855, 0.9120879120879121, 0.8461538461538461, 0.901098901098901, 0.9272727272727275, 0.9725274725274725, 0.6964285714285713, 0.8505494505494505, 0.914285714285714, 0.945054945054945, 0.9065934065934067, 0.8956043956043955, 0.9208791208791209, 0.9580419580419581, 0.9930069930069931, 1.0, 0.9230769230769231, 0.9510489510489512, 0.9300699300699302, 0.9370629370629372, 0.8764705882352942, 0.989010989010989, 0.9510489510489512, 0.8593406593406594, 0.8321428571428571, 0.9736263736263736, 0.8892857142857141, 0.9912087912087912, 0.8901098901098902, 0.9510489510489512, 0.9516483516483516, 0.8791208791208791, 0.6823529411764706, 0.8999999999999999, 0.8461538461538461, 0.9604395604395607, 0.956043956043956, 0.9632352941176472, 0.7892857142857141, 0.8214285714285712, 0.9725274725274725, 0.7307692307692307, 0.9285714285714285, 0.9393939393939393, 0.9272727272727275, 0.8461538461538463, 0.9757575757575757, 0.7500000000000001, 0.9300699300699302, 0.967032967032967, 0.8956043956043955, 0.8000000000000002, 0.8197802197802198, 0.9790209790209792, 0.9175824175824175, 0.6747252747252748, 0.8601398601398602, 1.0, 0.9076923076923077, 0.9065934065934067, 0.9964285714285712, 0.965034965034965, 0.9285714285714285, 0.9736263736263736, 0.8989010989010989, 0.9930069930069931, 0.9020979020979022, 0.8678571428571429, 0.9454545454545454, 0.9340659340659341, 0.8571428571428571, 0.881818181818182, 0.7186813186813187, 0.9941176470588234, 0.8391608391608393, 0.9945054945054945, 0.6678571428571428, 0.9725274725274725, 0.8928571428571428, 0.932142857142857, 0.8956043956043955, 0.9510489510489512, 0.9370629370629372, 0.8956043956043955, 0.8627450980392157, 0.9979360165118679, 0.9090909090909092, 0.85, 0.9252747252747253, 0.9860139860139862, 0.5804195804195805, 0.9181818181818183, 0.9208791208791209, 0.8186813186813187, 0.8626373626373626, 0.8749999999999999, 0.8881118881118882, 0.9930069930069931, 0.9727272727272729, 0.9120879120879121, 0.9020979020979022, 0.8671328671328673, 0.9912087912087912, 0.9384615384615385, 0.9930069930069931, 0.9440559440559443, 0.8956043956043955, 0.901098901098901, 0.9370629370629372, 0.989010989010989, 0.8999999999999999, 0.832967032967033, 1.0, 0.9300699300699302, 0.9720279720279721, 0.990909090909091, 0.9178571428571428, 0.9545454545454546, 0.7747252747252747, 0.7999999999999999, 0.6911764705882353, 0.8813186813186813, 0.9252747252747253, 0.9930069930069931, 0.9780219780219781, 0.8741258741258742, 0.8642857142857141, 0.911764705882353, 0.9736263736263736, 0.9930069930069931, 0.8945054945054945, 0.9300699300699302, 0.9395604395604396, 0.9607142857142855, 0.9505494505494506, 0.9727272727272729, 0.8956043956043955, 0.9835164835164836, 0.8811188811188813, 0.8901098901098902, 0.8681318681318682, 1.0, 0.9340659340659341, 0.989010989010989, 0.9428571428571427, 0.932142857142857, 0.9285714285714285, 0.9835164835164836, 1.0, 0.9790209790209792, 0.8285714285714285, 0.8791208791208791, 0.9428571428571428, 0.6703296703296703, 0.823529411764706, 0.9160839160839163, 0.9285714285714285, 0.945054945054945, 0.965034965034965, 0.8241758241758241, 0.9558823529411766, 0.7622377622377624, 0.9636363636363637, 0.9780219780219779, 0.7967032967032966, 0.9505494505494506, 0.9780219780219781, 0.9505494505494506, 0.945054945054945, 0.9588235294117649, 0.888235294117647, 0.9384615384615385, 0.9230769230769231, 0.9818181818181818, 0.9636363636363637, 0.7307692307692307, 0.711764705882353, 0.8749999999999999, 0.9780219780219779, 0.8470588235294119, 0.7252747252747254, 0.8676470588235294, 0.8284313725490197, 0.8499999999999999, 0.945054945054945, 0.8636363636363636, 0.8901098901098902, 0.9720279720279721, 0.9160839160839163, 0.9956043956043955, 0.8617131062951496, 0.9560439560439561, 0.9580419580419581, 0.8461538461538461, 0.982142857142857, 0.9785714285714284, 0.9285714285714285, 0.9340659340659341, 0.8461538461538461, 0.9692307692307693, 0.9035714285714285, 0.881818181818182, 0.95, 0.9752321981424149, 0.9285714285714285, 0.9464285714285712, 0.9065934065934067, 0.9720279720279721, 1.0, 0.9395604395604396, 0.8087719298245614, 0.9956043956043955, 0.965034965034965, 1.0, 0.9230769230769231, 0.8901098901098902, 0.95, 0.8558823529411764, 0.8956043956043955, 0.8406593406593407, 0.989010989010989, 0.8593406593406594, 0.9580419580419581, 0.9464285714285712, 0.9428571428571428, 0.9454545454545454, 0.9945054945054945, 0.8846153846153846, 0.9930069930069931, 0.9296703296703297, 0.9720279720279721]
# Create a histogram
plt.hist(data, bins='auto', alpha=0.7, color='blue')

# Add a title and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()
