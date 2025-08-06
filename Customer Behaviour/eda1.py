import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_eda(df,df_encoded):
    st.header("Exploratory Data Analysis", divider="gray")
    vis = st.selectbox("Choose Visualization", ("Correlation", "Box Plots for Monthly Charges", "Churn Rate by Contract Type", "Distribution of Customer Tenure",  "Churn Rate by Payment Method", "Churn Rate by Services", "Treemap of Churn Rate by Services"))

    if vis == "Correlation":
        with st.expander("Detailed Information", expanded=False):
            with st.expander("Equation Pearson", expanded=False):
                st.latex(r'''
                r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}
                ''')
                st.markdown('''Burgund, D., Nikolovski, S., Galić, D., & Maravić, N. (2023). Pearson Correlation in Determination of Quality of Current Transformers. Sensors (Basel, Switzerland), 23. https://doi.org/10.3390/s23052704.''')
            with st.expander("Analysis", expanded=False):
                st.markdown('''
1. Primary Churn Risk Factors:
    - Contract Type is the strongest protective factor (-0.397):
        - Month-to-month customers are more likely to churn
        - Long-term contracts significantly reduce churn risk
    - Customer Tenure has strong negative correlation (-0.352):
        - New customers are at higher risk of churning
        - Loyalty increases with tenure
                ''')
                st.markdown('''
2. Service Impact on Churn:
    - Security Services are crucial retention tools:
        - Online Security (-0.289) and Tech Support (-0.282) show strong negative correlations with churn
        - These services act as "sticky" features keeping customers engaged
    - Basic Services have less impact:
        - Phone Service and Internet Service alone don't significantly prevent churn
        - Value-added services are more important for retention
                ''')
                st.markdown('''
3. Pricing and Payment Insights:
    - Monthly Charges (0.193) slightly increases churn risk:
        - Higher bills correlate with increased churn
        - Multiple services (streaming, phone lines) increase monthly charges
    - Paperless Billing (0.192) shows positive correlation with churn:
        - May indicate less engaged or more transient customers
                ''')
                st.markdown('''
4. Customer Demographics:
    - Senior Citizens (0.151) show slightly higher churn tendency
    - Partners (-0.150) and Dependents (-0.164) reduce churn risk:
        - Family-connected customers are more stable
        - Single customers might need targeted retention strategies
                ''')
                st.markdown('''
5. Service Bundle Analysis:
    - Strong correlation between complementary services:
        - StreamingTV and StreamingMovies (0.435)
        - Multiple Lines and Monthly Charges (0.434)
    - Suggests successful cross-selling opportunities
                ''')
            with st.expander("Strategic Recommendations", expanded=False):
                st.markdown('''
1. Immediate Action Items:
    - Promote longer-term contracts with incentives
    - Enhance and actively market security and tech support services
    - Develop special retention programs for customers in first year
2. Customer Segment Focus:
    - Create family-oriented service bundles
    - Develop specific retention strategies for senior citizens
    - Target single customers with engagement programs
3. Service Enhancement:
    - Bundle security features with basic services
    - Offer tech support as part of standard packages
    - Create value-added service combinations
4. Pricing Strategies:
    - Review pricing for high-risk segments
    - Develop loyalty discounts based on tenure
    - Create bundle discounts for complementary services
5. Retention Program Development:
    - Focus on first-year customer experience
    - Implement early warning system for churn risk
    - Create upgrade paths to longer-term contracts
6. Long-term Strategic Focus:
    - Invest in technical support and security features
    - Develop family-oriented service packages
    - Create loyalty programs rewarding tenure
                ''')
        # Get only numeric columns
        numeric_columns = df_encoded.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df_encoded[numeric_columns].corr(method='pearson')

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            text=correlation_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu_r',
            zmid=0,
            showscale=True
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Correlation Matrix of Numeric Features',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=800,
            height=600,
            xaxis={
                'tickangle': 45,
                'tickfont': {'size': 10}
            },
            yaxis={
                'tickangle': 0,
                'tickfont': {'size': 10}
            },
            margin=dict(t=100, l=100, r=100, b=100)  # Adjust margins
        )

        # Add hover template
        fig.update_traces(
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{text}<extra></extra>"
        )

        # Display
        st.plotly_chart(fig, use_container_width=False)

        # Sort correlation values
        churn_correlation = df_encoded[numeric_columns].corr(method='pearson')['Churn'].drop('Churn')
        churn_correlation = churn_correlation.sort_values()

        # Determine colors based on correlation
        colors = ['red' if x < 0 else 'green' for x in churn_correlation.values]

        # Create the diverging bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=churn_correlation.index,
            x=churn_correlation.values,
            orientation='h',
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.7
        ))

        fig.update_layout(
            title={
                'text': 'Feature Correlation with Churn',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            xaxis_title='Correlation Coefficient',
            yaxis_title='Features',
            width=800,
            height=600,
            xaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgrey',
                griddash='dash',
                range=[-0.5, 0.5],
                dtick=0.2
            ),
            yaxis=dict(
                tickfont=dict(size=12)
            ),
            plot_bgcolor='white'
        )

        # Add a vertical line at x=0
        fig.add_vline(x=0, line_width=1, line_color="black")

        # Display
        st.plotly_chart(fig, use_container_width=False)

    elif vis == "Box Plots for Monthly Charges":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Average Spending Patterns:
    - Churned Customers (\$74.44) spend on average \$13.17 more than loyal customers (\$61.27)
    - This represents a ~21.5\% higher average monthly charge for churned customers
    - Clear indication that higher bills increase churn risk
2. Distribution Characteristics:
    - Median Comparison:
        - Loyal customers: \$64.43
        - Churned customers: \$79.65
        - The \$15.22 difference in medians suggests the price sensitivity threshold might be around \$70-75
    -   Spread Analysis:
        - Loyal customers show higher variability (std: \$31.09) compared to churned customers   (std: \$24.67)
        - Wider interquartile range for loyal customers (\$63.30) vs churned (\$38.05)
        - Suggests more price tolerance among loyal customers
3. Range Analysis:
    - Similar Price Ranges:
        - Loyal: \$18.25 - \$118.75 (range: \$100.50)
        - Churned: \$18.85 - \$118.35 (range: \$99.50)
    - Minimal difference in maximum charges indicates price alone isn't the sole factor
''')
        # Create box plot
        fig_charges = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            title='Monthly Charges Distribution by Churn Status',
            points='outliers',
            width=800,
            height=500
        )

        # Update layout
        fig_charges.update_layout(
            title={
                'text': 'Monthly Charges Distribution by Churn Status',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Churn Status",
            yaxis_title="Monthly Charges ($)",
            showlegend=False,
        )

        # Add mean line
        fig_charges.update_traces(boxmean=True)

        # Display
        st.plotly_chart(fig_charges, use_container_width=False)

    elif vis == "Churn Rate by Contract Type":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Contract Distribution Overview:
    - Total Customers: 7,043
    - Month-to-month: 3,875 (55.0%)
    - Two year: 1,695 (24.1%)
    - One year: 1,473 (20.9%)
            ''')    
            with st.expander("Detailed Analysis", expanded=False):
                st.markdown('''
1. Month-to-Month Contracts:
    - Highest Risk Segment (42.7% churn)
    - 1,655 customers churned out of 3,875
    - Critical metrics:
        - More than 4 in 10 customers leave
        - Represents majority of customer base (55%)
        - Retention rate of only 57.3%
2. One Year Contracts:
    - Moderate Risk Segment (11.3% churn)
    - 166 customers churned out of 1,473
    - Significant improvements:
        - 31.4% lower churn than month-to-month
        - Strong 88.7% retention rate
        - Shows value of longer commitment
3. Two Year Contracts:
    - Lowest Risk Segment (2.8% churn)
    - Only 47 customers churned out of 1,695
    - Best performance:
        - 97.2% retention rate
        - 8.5% improvement over one-year contracts
        - Demonstrates optimal contract length
                ''')
        # Calculate churn rates by contract type
        churn_by_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        
        # Reset index to make Contract a column
        churn_by_contract_reset = churn_by_contract.reset_index()
        
        # Melt the dataframe for proper plotting
        churn_by_contract_melted = churn_by_contract_reset.melt(
            id_vars=['Contract'],
            var_name='Churn_Status',
            value_name='Percentage'
        )
        
        # Create enhanced bar plot with custom colors
        fig_contract = px.bar(
            churn_by_contract_melted,
            x='Contract',
            y='Percentage',
            color='Churn_Status',
            color_discrete_map={'Yes': '#ef553b', 'No': '#636efa'},
            title='Churn Rate by Contract Type',
            barmode='group',
            width=800,
            height=500
        )
        
        # Update layout
        fig_contract.update_layout(
            title={
                'text': 'Churn Rate by Contract Type',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Contract Type",
            yaxis_title="Percentage (%)",
            yaxis_tickformat=',.1%'
        )
        
        st.plotly_chart(fig_contract, use_container_width=False)

    elif vis == "Distribution of Customer Tenure":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Comparison of Central Tendencies:
    - Loyal Customer (No Churn)
        - Mean tenure: 37.6 months
        - Median tenure: 38.0 months
        - Mode: 72.0 months
        - Nearly symmetric distribution (skewness: -0.032)
    - Churned Customers:
        - Mean tenure: 18.0 months
        - Median tenure: 10.0 months
        - Mode: 1.0 month
        - Highly right-skewed (skewness: 1.149)
2. Distribution Characteristics:
    - Spread Analysis:
        - Retained customers: Higher spread (std: 24.1 months)
        - Churned customers: Lower spread (std: 19.5 months)
        - Both groups max at 72 months
        - Retained minimum: 0 months
        - Churned minimum: 1 month
    - Quartile Analysis:
        - Retained Customers:
            - 25th percentile: 15 months
            - 75th percentile: 61 months
            - IQR: 46 months
        - Churned Customers:
            - 25th percentile: 2 months
            - 75th percentile: 29 months
            - IQR: 27 months
3. Critical Insights:
    - Early Churn Risk:
        - Mode of 1 month for churned customers indicates critical first month
        - 25% of churned customers leave within 2 months
        - 50% leave within 10 months
        - Early intervention window is crucial
    - Loyalty Development:
        - Mode of 72 months for retained customers shows strong loyalty potential
        - Even distribution around mean (37.6 months) for retained customers
        - Negative kurtosis (-1.414) indicates consistent spread across tenure periods
            ''')

        fig_tenure = px.histogram(
            df,
            x='tenure',
            color='Churn',
            color_discrete_map={'Yes': '#ef553b', 'No': '#636efa'},
            marginal='box',
            title='Customer Tenure Distribution by Churn Status',
            width=800,
            height=650
        )
        st.plotly_chart(fig_tenure, use_container_width=False)

    elif vis == "Churn Rate by Payment Method":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Electronic Check (Highest Risk Segment):
    - Represents largest customer base (33.58\%)
    - Alarmingly high churn rate (45.29\%)
    - Highest monthly charges for retained customers (\$74.23)
    - Critical Observation: Small price sensitivity (\$4.47 difference between churned and retained)
    - Action Required: This segment needs immediate intervention
2. Automatic Payment Methods (Most Stable):
    - **Bank Transfer**:
        - 21.92\% customer base
        - Excellent retention (83.29\%)
        - \$12.83 charge gap between churned/retained
    - **Credit Card**:
        - 21.61\% customer base
        - Best retention (84.76\%)
        - \$12.80 charge gap between churned/retained
    - **Success Pattern**: Both show similar stable patterns
3. Mailed Check (Most Price Sensitive):
    - 22.89\% customer base
    - Good retention (80.89\%)
    - Lowest average charges (\$41.40 retained)
    - Largest price sensitivity (\$13.16 difference)
    - Insight: Most price-conscious segment
            ''')
        payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
        fig_payment = px.bar(
            payment_churn,
            title='Churn Rate by Payment Method',
            barmode='group',
            color_discrete_map={'Yes': '#ef553b', 'No': '#636efa'},
        )
        st.plotly_chart(fig_payment, use_container_width=False)
    
    elif vis == "Churn Rate by Services":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Internet Service (Highest Impact):
    - **Critical Issue**: Fiber optic has 41.89% churn rate vs DSL's 18.96%
    - **Opportunity**: Customers with no internet service have only 7.40% churn
    - **Key Insight**: While fiber optic has highest adoption (43.96%), it's also bleeding customers the most
    - **Action Items**:
        - Investigate fiber optic service quality and pricing
        - Consider DSL as a stable service offering
        - Analyze why no-internet customers are more loyal
2. Security Services (Strong Retention Effect):
    - **OnlineSecurity**:
        - Without: 41.77% churn
        - With: 14.61% churn (27.16% difference)
    - **TechSupport**:
        - Without: 41.64% churn
        - With: 15.17% churn (26.47% difference)
    - **Action Items**:
        - Prioritize these services for upselling
        - Only 28.67% have OnlineSecurity and 29.02% have TechSupport - huge growth potential
3. Backup & Protection Services (Moderate Impact):
    - **OnlineBackup**:
        - Without: 39.93% churn
        - With: 21.53% churn
    - **DeviceProtection**:
        - Without: 39.13% churn
        - With: 22.50% churn
    - **Action Items**:
        - Bundle these services with security offerings
        - Target the ~44% of customers without these services
4. Entertainment Services (Minimal Impact):
    - **StreamingTV & Movies**:
        - Similar patterns: ~30% churn with service, ~33% without
        - Good adoption rates: ~38% for both services
    - **Action Items**:
        - Not primary focus for churn reduction
        - Consider bundling with high-impact services
5. Phone Services (Low Impact):
    - **Basic Phone Service**:
        - High adoption (90.32%) but minimal churn difference
    - **Multiple Lines**:
        - Slightly higher churn with multiple lines (28.61% vs 25.04%)
    - **Action Items**:
        - Focus on upselling internet and security services instead
            ''')
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        churn_rates = []
        for service in service_columns:
            churn_rate = df.groupby(service)['Churn'].value_counts(normalize=True).unstack()
            churn_rates.append(churn_rate)

        # Create subplots for services
        fig_services = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=service_columns
        )

        for idx, (service, churn_rate) in enumerate(zip(service_columns, churn_rates)):
            row = idx // 3 + 1
            col = idx % 3 + 1

            fig_services.add_trace(
                go.Bar(x=churn_rate.index, y=churn_rate['Yes'], name=service),
                row=row,
                col=col
            )

        fig_services.update_layout(
            height=1000,
            title_text="Churn Rate by Services",
            showlegend=False
        )

        st.plotly_chart(fig_services, use_container_width=False)

    elif vis == "Treemap of Churn Rate by Services":
        with st.expander("Analysis", expanded=False):
            st.markdown('''
1. Core Internet Services (Highest Impact):
    - Fiber Optic: 41.89\% churn (43.96\% base)
    - DSL: 18.96\% churn (34.37\% base)
    - No Internet: 7.40\% churn (21.67\% base)
    - Impact Range: 34.49\% (Fiber vs No Internet)
    - Critical Finding: Internet service type is the strongest predictor of churn
2. Security & Support Tier (High Impact):
    - Online Security:
        - Without: 41.77\% churn (49.67\% base)
        - With: 14.61\% churn (28.67\% base)
        - Impact: 27.16\% reduction

    - Tech Support:
        - Without: 41.64\% churn (49.31\% base)
        - With: 15.17\% churn (29.02\% base)
        - Impact: 26.47\% reduction
    
    - Key Insight: Security and support services have nearly identical churn reduction effects
3. Protection Services Tier (Medium Impact):
    - Online Backup:
        - Without: 39.93\% churn (43.84\% base)
        - With: 21.53\% churn (34.49\% base)
        - Impact: 18.40\% reduction

    - Device Protection:
        - Without: 39.13\% churn (43.94\% base)
        - With: 22.50\% churn (34.39\% base)
        - Impact: 16.63\% reduction
    
    - Pattern: Similar adoption rates and churn reduction effects
4. Entertainment Services (Low Impact):
    - Streaming TV:
        - Without: 33.52\% churn (39.90\% base)
        - With: 30.07\% churn (38.44\% base)
        - Impact: 3.45\% reduction

    - Streaming Movies:
        - Without: 33.68\% churn (39.54\% base)
        - With: 29.94\% churn (38.79\% base)
        - Impact: 3.74\% reduction
    
    - Note: Minimal impact on churn reduction
5. Phone Services (Minimal Impact):
    - Basic Phone:
        - Without: 24.93\% churn (9.68\% base)
        - With: 26.71\% churn (90.32\% base)
        - Impact: -1.78\% (slightly increases churn)

    - Multiple Lines:
        - Without: 25.04\% churn (48.13\% base)
        - With: 28.61\% churn (42.18\% base)
        - Impact: -3.57\% (increases churn)
            ''')
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Create a list to store the data for treemap
        treemap_data = []

        for service in service_columns:
            # Calculate churn rates for each service category
            churn_by_service = df.groupby([service, 'Churn']).size().reset_index(name='count')
            total_by_service = churn_by_service.groupby(service)['count'].sum().reset_index()

            # Merge total counts with churn counts
            churn_by_service = churn_by_service.merge(total_by_service, on=service, suffixes=('', '_total'))
            churn_by_service['percentage'] = (churn_by_service['count'] / churn_by_service['count_total'] * 100).round(2)

            # Add to treemap data
            for _, row in churn_by_service.iterrows():
                treemap_data.append({
                    'Service': 'Services',  # Root level
                    'Type': service,        # Service type
                    'Category': f"{row[service]}",  # Service category
                    'Status': row['Churn'],  # Churn status
                    'Count': row['count'],
                    'Percentage': row['percentage']
                })

        # Create DataFrame for treemap
        df_treemap = pd.DataFrame(treemap_data)

        # Create treemap
        fig = px.treemap(
            df_treemap,
            path=['Service', 'Type', 'Category', 'Status'],
            values='Count',
            color='Percentage',
            color_continuous_scale='RdYlGn',  # Red for high churn, green for low churn
            title='Service Usage and Churn Rate Distribution',
            custom_data=['Percentage']
        )

        # Update layout
        fig.update_layout(
            width=1200,
            height=800,
            title={
                'text': 'Service Usage and Churn Rate Distribution',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )

        # Update hover template
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>" +
                          "Count: %{value}<br>" +
                          "Churn Rate: %{customdata[0]:.1f}%<extra></extra>"
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=False)