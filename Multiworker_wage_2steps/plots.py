import matplotlib.pyplot as plt
import os

class Plots:

    def __init__(self, p):

        # Create a directory to save plots
        self.output_dir = 'Plots'
        if not os.path.exists(self.output_dir):
         os.makedirs(self.output_dir)
        self.p = p

    def CRSvsDRSvalue(self, cc_W, cc_J, mwc_W=None, mwc_J=None, mwc_s_W=None, mwc_s_J=None, mwc_s_dir_W=None, mwc_s_dir_J=None, save=0):
        # Create a figure with a specific size
        plt.figure(figsize=(16, 6))  # Width=16 inches, Height=6 inches

        # First subplot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        # Plot the data
        plt.plot(cc_W[self.p.z_0-1, :], cc_J[self.p.z_0-1, :], label='CRS')
        if mwc_W is not None:
            plt.plot(mwc_W[self.p.z_0-1, 0, 1, :, 1], mwc_J[self.p.z_0-1, 0, 1, :], label='DRS')
        if mwc_s_W is not None:
            plt.plot(mwc_s_W[self.p.z_0-1, 0, 1, :, 1], mwc_s_J[self.p.z_0-1, 0, 1, :], label='DRS with separations')
        if mwc_s_dir_W is not None:
            plt.plot(mwc_s_dir_W[self.p.z_0-1, 0, 1, :, 1], mwc_s_dir_J[self.p.z_0-1, 0, 1, :], label='DRS with direct separations')
        # Add titles and labels
        plt.title('Value across models, 1 senior worker')
        plt.xlabel('Worker value')
        plt.ylabel('Job value')

        plt.legend()

        #plt.ylim([0, 200])

        # Second subplot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(cc_W[self.p.z_0-1, :], cc_J[self.p.z_0-1, :], label='CRS')
        if mwc_W is not None:
            plt.plot(mwc_W[self.p.z_0-1, 0, 1, :, 1], mwc_J[self.p.z_0-1, 0, 1, :], label='DRS')
        if mwc_s_W is not None:
            plt.plot(mwc_s_W[self.p.z_0-1, 0, 1, :, 1], mwc_s_J[self.p.z_0-1, 0, 1, :], label='DRS with separations')
        if mwc_s_dir_W is not None:
            plt.plot(mwc_s_dir_W[self.p.z_0-1, 0, 1, :, 1], mwc_s_dir_J[self.p.z_0-1, 0, 1, :], label='DRS with direct separations')

        # Add titles and labels
        plt.title('Value across models, 1 senior worker zoomed in')
        plt.xlabel('Worker value')
        plt.ylabel('Job value')

        plt.ylim([0, 50])

        plt.legend()


        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add legend
        plt.legend()

        # Save the plot to a file in the specified directory
        if save==1:
            plt.savefig(os.path.join(self.output_dir,'Value CRS vs DRS.png'), bbox_inches='tight')  # Save as PNG


        # Optionally set x and y limits
        #plt.xlim([-10, 40])
        #plt.ylim([0, 200])
        # Save the plot to a file
        if save==1:
            plt.savefig(os.path.join(self.output_dir,'Value CRS vs DRS zoom.png'), bbox_inches='tight')  # Save as PNG

        # Show the plot (optional)
        plt.show()

    def CRSvsDRSpolicy(self, cc_W, cc_Wstar, mwc_W=None, mwc_Wstar=None, mwc_s_W=None, mwc_s_Wstar=None, mwc_s_dir_W=None, mwc_s_dir_Wstar=None, save=0):
        plt.figure(figsize=(8, 6))  # Width=16 inches, Height=12 inches

        # Plot the data
        plt.plot(cc_W[self.p.z_0-1, :], cc_Wstar[self.p.z_0-1, :]-cc_W[self.p.z_0-1, :], label='CRS')
        if mwc_W is not None:
            plt.plot(mwc_W[self.p.z_0-1, 0, 1,:, 1], mwc_Wstar[self.p.z_0-1, 0, 1, :]-mwc_W[self.p.z_0-1, 0, 1,:, 1], label='DRS')
        if mwc_s_W is not None:
            plt.plot(mwc_s_W[self.p.z_0-1, 0, 1,:, 1], mwc_s_Wstar[self.p.z_0-1, 0, 1, :]-mwc_s_W[self.p.z_0-1, 0, 1,:, 1], label='DRS with separations')
        if mwc_s_dir_W is not None:
            plt.plot(mwc_s_dir_W[self.p.z_0-1, 0, 1,:, 1], mwc_s_dir_Wstar[self.p.z_0-1, 0, 1, :]-mwc_s_dir_W[self.p.z_0-1, 0, 1,:, 1], label='DRS with direct separations')
        plt.legend()
        # Add titles and labels
        plt.title('Future value across types')
        plt.xlabel('Worker value')
        plt.ylabel('Value change over time')
        if save==1:
          plt.savefig(os.path.join(self.output_dir,'Policy CRS vs DRS.png'), bbox_inches='tight')  # Save as PNG

    def ValueComparison(self, jun1,sen1,jun2,sen2 ,cc_W, cc_J, mwc_W=None, mwc_J=None, mwc_s_W=None, mwc_s_J=None, mwc_s_dir_W=None, mwc_s_dir_J=None, save=0):
        # Create a figure with a specific size
        plt.figure(figsize=(16, 6))  # Width=16 inches, Height=6 inches

        # First subplot
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        # Plot the data
        jun=jun1
        sen=sen1
        if (jun==0) & (sen==1):
            plt.plot(cc_W[self.p.z_0-1, :], cc_J[self.p.z_0-1, :], label='CRS') 
        if mwc_W is not None:
            plt.plot(mwc_W[self.p.z_0-1, jun, sen, :, 1], mwc_J[self.p.z_0-1, jun, sen, :], label='DRS')
        if mwc_s_W is not None:        
            plt.plot(mwc_s_W[self.p.z_0-1, jun, sen, :, 1], mwc_s_J[self.p.z_0-1, jun, sen, :], label='DRS with sep')
        if mwc_s_dir_W is not None:        
            plt.plot(mwc_s_dir_W[self.p.z_0-1, jun, sen, :, 1], mwc_s_dir_J[self.p.z_0-1, jun, sen, :], label='DRS with direct sep')
        plt.title(f'Value across models, {jun} juniors and {sen} seniors')
        plt.xlabel('Worker value')
        plt.ylabel('Job value')
        plt.legend()

        plt.legend()

        #plt.ylim([0, 200])

        # Second subplot
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        jun=jun2
        sen=sen2
        if (jun==0) & (sen==1):
            plt.plot(cc_W[self.p.z_0-1, :], cc_J[self.p.z_0-1, :], label='CRS')        
        if mwc_W is not None:
            plt.plot(mwc_W[self.p.z_0-1, jun, sen, :, 1], mwc_J[self.p.z_0-1, jun, sen, :], label='DRS')
        if mwc_s_W is not None:      
            plt.plot(mwc_s_W[self.p.z_0-1, jun, sen, :, 1], mwc_s_J[self.p.z_0-1, jun, sen, :], label='DRS with sep')
        if mwc_s_dir_W is not None:
            plt.plot(mwc_s_dir_W[self.p.z_0-1, jun, sen, :, 1], mwc_s_dir_J[self.p.z_0-1, jun, sen, :], label='DRS with direct sep')

        plt.title(f'Value across models, {jun} juniors and {sen} seniors')
        plt.xlabel('Worker value')
        plt.ylabel('Job value')

        #plt.ylim([0, 50])

        plt.legend()


        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add legend
        plt.legend()

        # Save the plot to a file in the specified directory
        if save==1:
         plt.savefig(os.path.join(self.output_dir,'Value CRS vs DRS.png'), bbox_inches='tight')  # Save as PNG


        # Optionally set x and y limits
        #plt.xlim([-10, 40])
        #plt.ylim([0, 200])
        # Save the plot to a file
        if save==1:
          plt.savefig(os.path.join(self.output_dir,'Value CRS vs DRS zoom.png'), bbox_inches='tight')  # Save as PNG

        # Show the plot (optional)
        plt.show()
    
    def CRSvsDRSdirect(self, cc_J, mwc_J=None, save=0):

        plt.figure(figsize=(8, 6))  # Width=16 inches, Height=12 inches

        # Plot the data
        if mwc_J is not None:
            plt.plot(cc_J[self.p.z_0-1, :], mwc_J[self.p.z_0-1, 0, 1, :], label='CRS vs DRS')
        if save==1:
          plt.savefig(os.path.join(self.output_dir,'Value CRS vs DRS direct.png'), bbox_inches='tight')  # Save as PNG
        
        plt.title(f'Value across models, Direct comparison (across wage levels)')
        plt.xlabel('CRS Value function')
        plt.ylabel('DRS Value function')

    def CRSvsDRSsurplus(self, cc_J, cc_W ,mwc_J=None, mwc_W=None, save=0):
        fig, ax = plt.subplots(figsize=(8, 6))#This creates a figure and a set of subplots with a specific size. The size is specified in inches (width, height).

        # Plot the original data
        ax.plot(cc_W[self.p.z_0-1,:]+cc_J[self.p.z_0-1,:], mwc_W[self.p.z_0-1, 0, 1,:,1]+mwc_J[self.p.z_0-1, 0, 1, :], label='Total surplus')

        # Calculate the range for the 45-degree line
        x_vals = cc_W[self.p.z_0-1,:] + cc_J[self.p.z_0-1,:]
        y_vals = mwc_W[self.p.z_0-1, 0, 1,:,1] + mwc_J[self.p.z_0-1, 0, 1, :]
        min_val = min(min(x_vals), min(y_vals))
        max_val = max(max(x_vals), max(y_vals))

        # Add the 45-degree line
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='45-degree line')

        # Add titles and labels
        ax.set_title('Comparing total surplus across methods')
        ax.set_xlabel('Total surplus (CRS)')
        ax.set_ylabel('Total surplus (DRS)')
        ax.legend()

        # Save the plot to a file
        if save==1:
            plt.savefig(os.path.join(self.output_dir,'Surplus CRS vs DRS.png'), bbox_inches='tight')  # Save as PNG

        # Show the plot
        plt.show()