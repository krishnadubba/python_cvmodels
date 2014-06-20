from mle_cat import mle_cat

# Generate random values from the categorical distribution 
# with 6 categories and the corresponding probabilities.
original_probabilities = [0.25, 0.15, 0.1, 0.1, 0.15, 0.25];
r1 = randsample (6, 200, true, original_probabilities);
r2 = randsample (6, 20000, true, original_probabilities);

# Estimate the categorical distribution parameters from the data.
estimated_probabilities1 = mle_cat(r1, 6)
estimated_probabilities2 = mle_cat(r2, 6)

# Plot the original and the estimated models for comparison.
subplot(1,3,1);
bar(original_probabilities, 'b');
axis([0,7,0,0.3]);
xlabel('x', 'FontSize', 16);
ylabel('P(x)', 'FontSize', 16);
set(gca,'YTick',[0, 0.25]);
title('(a)', 'FontSize', 14);

subplot(1,3,2);
bar(estimated_probabilities1, 'r');
axis([0,7,0,0.3]);
xlabel('x', 'FontSize', 16);
set(gca,'YTick',[0, 0.25]);
title('(b)', 'FontSize', 14);

subplot(1,3,3);
bar(estimated_probabilities2, 'r');
axis([0,7,0,0.3]);
xlabel('x', 'FontSize', 16);
set(gca,'YTick',[0, 0.25]);
title('(c)', 'FontSize', 14);