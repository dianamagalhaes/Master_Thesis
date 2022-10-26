#python classe_ataque_cleverhans_v4.0.0.py -m Demo --device cuda:0 eval --attack Fast_Gradient_Method
#python classe_ataque_cleverhans_v4.0.0.py -m Demo --device cuda:0 eval --attack Projected_Gradient_Descent
#python classe_ataque_cleverhans_v4.0.0.py -m Demo --device cuda:0 eval --attack Sparse_L1_Descent
# python classe_ataque_cleverhans_v4.0.0.py -m Demo --device cuda:0 eval --attack Carlini_Wagner_L2
#python classe_ataque_cleverhans_v4.0.0.py -m Demo --device cuda:0 eval --attack Hop_Skip_Jump

#python classe_ataque_tf.py -m SA_Classification_AI4MED --device cuda:0 --attack_name Fast_Gradient_Method
python classe_ataque_tf.py -m SA_Classification_AI4MED --device cuda:0 --attack_name Carlini_Wagner_L2
#python classe_ataque_tf.py -m SA_Classification_AI4MED --device cuda:0 --attack_name Projected_Gradient_Descent
