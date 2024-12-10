twins = dlmread([datadir '/twins.txt']);
twins = twins(2:end,2:end);
grotKEEP = true(size(all_vars,1),1);
grotKEEP(find(all_vars(:,1)==376247))=0;
grotKEEP(find(all_vars(:,1)==168240))=0;
grotKEEP(find(all_vars(:,1)==122418))=0;
twins = twins(grotKEEP,grotKEEP);