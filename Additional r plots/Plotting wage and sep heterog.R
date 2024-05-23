install.packages("ggplot2")
library(ggplot2)
#Making wage and separations cyclicality graphs! Manually!!!
#Separations cyclicality across tenure and size
x<-1:3
y1<-c(0.18,0.39,0.24) #cyclicality of young workers
y2<-c(0.07,-0.04,-0.06) #cyclicality of workers with 10 years of tenure

#Or interacting with occupation heterogeneity
x<-1:3
y1<-c(0.099,0.18,0.06)
y2<-c(0.003,-0.19,-0.2)
data <- data.frame(x, y1,y2)
# Create the plot
plot <- ggplot(data) +
  geom_line(aes(x, y1, color="Young workers")) +
  geom_point(aes(x, y1, color="Young workers"),size=2) +
  geom_line(aes(x, y2, color="Workers with 10y tenure")) +
  geom_point(aes(x, y2, color="Workers with 10y tenure"),size=2) +
  scale_x_continuous(breaks=1:10)+
  labs( x = "Firm size brackets",y="Separation cyclicality",color="Tenure at the establishment") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        #panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 

# Display the plot
print(plot)

ggsave("Separations Cyclicality.jpg", plot, width = 10, height = 4, units = "in", dpi = 300, type = "cairo")


#Wage cyclicality across tenure and size
x<-1:3
y1<-c(-1.22,-1.20,-0.95) #cyclicality of young workers
y2<-c(-1.42,-1.40,-1.14) #cyclicality of worker with 10 years of tenure

data <- data.frame(x, y1,y2)
# Create the plot
plot <- ggplot(data) +
  geom_line(aes(x, y1, color="Young workers")) +
  geom_point(aes(x, y1, color="Young workers"),size=2) +
  geom_line(aes(x, y2, color="Workers with 10y tenure")) +
  geom_point(aes(x, y2, color="Workers with 10y tenure"),size=2) +
  scale_x_continuous(breaks=1:10)+
  labs( x = "Firm size brackets",y="Wage cyclicality",color="Tenure at the establishment") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        #panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank()) 

# Display the plot
print(plot)

ggsave("Wage Cyclicality.jpg", plot, width = 10, height = 4, units = "in", dpi = 300, type = "cairo")
