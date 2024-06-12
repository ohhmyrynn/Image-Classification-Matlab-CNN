% Misalkan 'confusion_matrix' adalah matriks konfussi 4x4 Anda
confusion_matrix = magic(4);
classLabels = {'Class1', 'Class2', 'Class3', 'Class4'};
confusionchart(confusion_matrix, classLabels);
% Hitung total prediksi dan aktual positif
total_positives = sum(sum(confusion_matrix));
actual_positives = sum(confusion_matrix(:, 4)); %ekstraks kolom keempat
predicted_positives = sum(confusion_matrix(4, :)); %ekstrakks baris keempat

% Hitung akurasi
accuracy = sum(diag(confusion_matrix)) / total_positives;

% Hitung presisi untuk setiap kelas (kolom)
precision = zeros(1, 4);
for i = 1:4
    precision(i) = confusion_matrix(i, i) / sum(confusion_matrix(:, i));
end

% Hitung recall untuk setiap kelas (baris)
recall = zeros(1, 4);
for i = 1:4
    recall(i) = confusion_matrix(i, i) / sum(confusion_matrix(i, :));
end

% Hitung F1-score untuk setiap kelas
f1_score = zeros(1, 4);
for i = 1:4
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Hitung rata-rata presisi dan recall
average_precision = mean(precision);
average_recall = mean(recall);

% Hitung F1-score rata-rata
f1_score_average = 2 * (average_precision * average_recall) / (average_precision + average_recall);

% Tampilkan hasil
disp('Akurasi:');
disp(accuracy);

disp('Presisi:');
disp(precision);

disp('Recall:');
disp(recall);

disp('F1-score:');
disp(f1_score);

disp('Rata-rata Presisi:');
disp(average_precision);

disp('Rata-rata Recall:');
disp(average_recall);

disp('Rata-rata F1-score:');
disp(f1_score_average);
