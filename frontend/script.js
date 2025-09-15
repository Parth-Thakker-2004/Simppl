class ChatApp {
    constructor() {
        this.isInitialized = false;
        this.messages = [];
        this.settings = {
            platformFilter: '',
            useOnline: true,
            topK: 5
        };
        
        this.initializeElements();
        this.bindEvents();
        this.initializeTheme();
        this.closeSettingsOnClickOutside();
        this.initializeSystem();
    }

    initializeElements() {
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.themeToggle = document.getElementById('themeToggle');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.welcomeMessage = document.getElementById('welcomeMessage');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.contextIndicator = document.getElementById('contextIndicator');
        this.clearConversationBtn = document.getElementById('clearConversationBtn');
        this.exportPdfBtn = document.getElementById('exportPdfBtn');
        
        // Settings elements
        this.platformFilter = document.getElementById('platformFilter');
        this.useOnline = document.getElementById('useOnline');
        this.topK = document.getElementById('topK');
        this.topKValue = document.getElementById('topKValue');
    }

    bindEvents() {
        // Send button and Enter key
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Input changes
        this.messageInput.addEventListener('input', () => this.updateSendButton());

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Settings
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        this.platformFilter.addEventListener('change', () => this.updateSettings());
        this.useOnline.addEventListener('change', () => this.updateSettings());
        this.topK.addEventListener('input', () => {
            this.topKValue.textContent = this.topK.value;
            this.updateSettings();
        });
        
        // Clear conversation
        if (this.clearConversationBtn) {
            this.clearConversationBtn.addEventListener('click', () => this.clearConversation());
        }
        
        // Export PDF
        if (this.exportPdfBtn) {
            this.exportPdfBtn.addEventListener('click', () => this.exportResearchPaper());
        }

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const icon = this.themeToggle.querySelector('.theme-icon');
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    toggleSettings() {
        this.settingsPanel.classList.toggle('open');
    }
    
    closeSettingsOnClickOutside() {
        // Close settings when clicking outside the panel
        document.addEventListener('click', (event) => {
            // If settings is open and the click is outside the settings panel and settings button
            if (this.settingsPanel.classList.contains('open') && 
                !this.settingsPanel.contains(event.target) && 
                !this.settingsBtn.contains(event.target)) {
                this.settingsPanel.classList.remove('open');
            }
        });
    }

    updateSettings() {
        this.settings = {
            platformFilter: this.platformFilter.value,
            useOnline: this.useOnline.checked,
            topK: parseInt(this.topK.value)
        };
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || !this.isInitialized;
    }

    async initializeSystem() {
        this.showLoading('Initializing RAG system...');
        
        try {
            const response = await fetch('http://localhost:5000/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    gemini_api_key: 'AIzaSyCvWysEM6_RGTf3jPtlZLEAMPbaZjUBjwY'
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.isInitialized = true;
                this.updateSendButton();
                this.hideLoading();
                this.showNotification('System initialized successfully!', 'success');
            } else {
                throw new Error(data.message || 'Failed to initialize');
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.hideLoading();
            this.showNotification('Failed to initialize system. Please check if the backend is running.', 'error');
        }
    }

    showLoading(message = 'Loading...') {
        this.loadingOverlay.querySelector('p').textContent = message;
        this.loadingOverlay.classList.add('show');
    }

    hideLoading() {
        this.loadingOverlay.classList.remove('show');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: type === 'error' ? '#f44336' : type === 'success' ? '#4caf50' : '#2196f3',
            color: 'white',
            padding: '1rem 1.5rem',
            borderRadius: '8px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15)',
            zIndex: '1001',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    async sendMessage() {
        const text = this.messageInput.value.trim();
        if (!text || !this.isInitialized) return;

        // Hide welcome message
        if (this.welcomeMessage) {
            this.welcomeMessage.style.display = 'none';
        }

        // Add user message
        this.addMessage(text, 'user');
        this.messageInput.value = '';
        this.updateSendButton();
        this.autoResizeTextarea();

        // Show typing indicator
        const typingId = this.addTypingIndicator();
        
        // Show context indicator after first message (when context is being used)
        if (this.messages.length > 1) {
            this.contextIndicator.classList.add('active');
        }

        try {
            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: text,
                    platform_filter: this.settings.platformFilter || null,
                    use_online: this.settings.useOnline,
                    top_k: this.settings.topK
                })
            });

            const data = await response.json();

            // Remove typing indicator
            this.removeTypingIndicator(typingId);

            if (data.response) {
                this.addMessage(data.response, 'assistant', data.sources, data.graphs);
            } else {
                throw new Error(data.error || 'Unknown error');
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.removeTypingIndicator(typingId);
            this.addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'assistant');
        }
    }

    addMessage(content, role, sources = null, graphs = null) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;

        const avatar = role === 'user' ? 'U' : 'AI';
        const roleName = role === 'user' ? 'You' : 'Assistant';

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = this.generateSourcesHtml(sources);
        }
        
        // Generate graphs HTML if available
        let graphsHtml = '';
        if (graphs && graphs.length > 0) {
            graphsHtml = this.generateGraphsHtml(graphs);
        }

        messageElement.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">${avatar}</div>
                <div class="message-role">${roleName}</div>
            </div>
            <div class="message-content">
                ${this.formatMessage(content)}
                ${graphsHtml}
                ${sourcesHtml}
            </div>
        `;

        this.chatContainer.appendChild(messageElement);
        this.scrollToBottom();

        // Store message
        this.messages.push({ content, role, sources, timestamp: Date.now() });
    }

    addTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.className = 'message assistant typing';
        const typingId = 'typing-' + Date.now();
        typingElement.id = typingId;

        typingElement.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">AI</div>
                <div class="message-role">Assistant</div>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        // Add typing animation styles
        const style = document.createElement('style');
        style.textContent = `
            .typing-indicator {
                display: flex;
                gap: 0.25rem;
                align-items: center;
            }
            .typing-indicator span {
                width: 6px;
                height: 6px;
                background: var(--text-secondary);
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
                30% { transform: translateY(-10px); opacity: 1; }
            }
        `;
        document.head.appendChild(style);

        this.chatContainer.appendChild(typingElement);
        this.scrollToBottom();

        return typingId;
    }

    removeTypingIndicator(typingId) {
        const element = document.getElementById(typingId);
        if (element) {
            element.remove();
        }
    }

    generateSourcesHtml(sources) {
        if (!sources || sources.length === 0) return '';

        const socialSources = sources.filter(s => s.type === 'social_media');
        const newsSources = sources.filter(s => s.type === 'news');

        let html = '<div class="sources-section">';
        html += '<div class="sources-header">📚 Sources</div>';
        
        // Display news sources first with more prominence
        if (newsSources.length > 0) {
            html += '<div class="source-category news-category">📰 News Sources</div>';
            newsSources.forEach(source => {
                const formattedDate = source.publication_date ? 
                    new Date(source.publication_date).toLocaleDateString('en-US', {
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric'
                    }) : '';
                
                html += `
                    <div class="source-item news-item">
                        <div class="source-badge">News</div>
                        <div class="source-header">
                            <a href="${source.url}" target="_blank" class="source-link">${source.title}</a>
                        </div>
                        <div class="source-preview">${source.snippet}</div>
                        <div class="source-metadata">
                            ${source.publisher ? `<div class="source-publisher">📝 <a href="${source.url}" target="_blank" class="source-publisher-link">${source.publisher}</a></div>` : ''}
                            ${formattedDate ? `<div class="source-date">📅 ${formattedDate}</div>` : ''}
                        </div>
                    </div>
                `;
            });
        }

        if (socialSources.length > 0) {
            html += '<div class="source-category">💬 Social Media</div>';
            socialSources.slice(0, 3).forEach(source => {
                html += `
                    <div class="source-item">
                        <div class="source-header">
                            <span class="source-type">
                                ${source.url ? 
                                    `<a href="${source.url}" target="_blank" class="source-platform-link">${source.platform}</a>` : 
                                    source.platform
                                }
                            </span>
                            <span class="source-platform">
                                ${source.content_type} by 
                                ${source.url ? 
                                    `<a href="${source.url}" target="_blank" class="source-username-link">${source.username}</a>` : 
                                    source.username
                                }
                            </span>
                        </div>
                        <div class="source-preview">${source.preview}</div>
                    </div>
                `;
            });
        }

        html += '</div>';
        return html;
    }

    generateGraphsHtml(graphs) {
        if (!graphs || graphs.length === 0) return '';
        
        // Select only one random graph if multiple are available
        let selectedGraph;
        if (graphs.length > 1) {
            // Choose a random graph
            const randomIndex = Math.floor(Math.random() * graphs.length);
            selectedGraph = graphs[randomIndex];
        } else {
            // Just use the only graph we have
            selectedGraph = graphs[0];
        }
        
        let html = '<div class="graphs-container">';
        html += '<div class="graphs-header">📊 Data Insight</div>';
        
        // Handle both formats: direct base64 or data URL
        let imgSrc = '';
        if (selectedGraph.image_data) {
            // Direct base64 format
            imgSrc = `data:image/png;base64,${selectedGraph.image_data}`;
        } else if (selectedGraph.image) {
            // Data URL format (already contains the prefix)
            imgSrc = selectedGraph.image;
        }
        
        html += `
            <div class="graph-item">
                <div class="graph-title">${selectedGraph.title}</div>
                <div class="graph-image">
                    <img src="${imgSrc}" alt="${selectedGraph.title}">
                </div>
                <div class="graph-description">${selectedGraph.description}</div>
            </div>
        `;
        
        html += '</div>';
        return html;
    }

    formatMessage(content) {
        // Simple markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        }, 100);
    }
    
    clearConversation() {
        // Show a confirmation dialog
        if (!confirm("Are you sure you want to clear the entire conversation history?")) {
            return; // User cancelled
        }
        
        // Clear conversation history on the server
        fetch('http://localhost:5000/api/conversation/clear', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success || data.message) {
                // Clear the UI
                this.chatContainer.innerHTML = '';
                
                // Show welcome message again
                if (this.welcomeMessage) {
                    this.welcomeMessage.style.display = 'flex';
                }
                
                // Hide context indicator
                this.contextIndicator.classList.remove('active');
                
                // Clear local message history
                this.messages = [];
                
                // Show confirmation
                this.showToast('Conversation cleared successfully!');
                
                // Close settings panel
                if (this.settingsPanel.classList.contains('active')) {
                    this.toggleSettings();
                }
            }
        })
        .catch(error => {
            console.error('Error clearing conversation:', error);
            this.showToast('Failed to clear conversation. Please try again.');
        });
    }
    
    showToast(message) {
        // Remove any existing toasts
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => {
            toast.classList.remove('show');
            setTimeout(() => {
                toast.remove();
            }, 300);
        });
        
        // Create new toast
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Show toast with slight delay
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        
        // Hide and remove toast after a delay
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                toast.remove();
            }, 300); // Wait for transition to complete before removing
        }, 3000); // Show for 3 seconds
    }
    
    exportResearchPaper() {
        // Check if we have messages to export
        if (this.messages.length === 0) {
            this.showToast('No conversation to export');
            return;
        }
        
        // Show a loading toast
        this.showToast('Generating research paper...');
        
        // Get current date for the paper
        const currentDate = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        // Extract a title from the first user message
        const firstUserMessage = this.messages.find(msg => msg.role === 'user');
        const title = firstUserMessage ? 
            firstUserMessage.content.split('.')[0].substring(0, 100) + 
            (firstUserMessage.content.length > 100 ? '...' : '') : 
            'Journalistic Analysis';
            
        // Create paper sections
        const abstract = this.generateAbstract();
        const introduction = this.generateIntroduction();
        const methodology = this.generateMethodology();
        const results = this.generateResults();
        const discussion = this.generateDiscussion();
        const conclusion = this.generateConclusion();
        const references = this.generateReferences();
        
        setTimeout(() => {
            try {
                // Initialize jsPDF with proper academic paper margins
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF({
                    orientation: 'portrait',
                    unit: 'mm',
                    format: 'a4'
                });
                
                // Set document properties
                doc.setProperties({
                    title: title,
                    subject: 'Journalistic Analysis',
                    author: 'Social Media Assistant',
                    keywords: 'journalism, social media analysis, news',
                    creator: 'Social Media Assistant'
                });
                
                // Set font styles
                doc.setFont('helvetica', 'normal');
                
                // Title
                doc.setFontSize(20);
                doc.setFont('helvetica', 'bold');
                const titleLines = doc.splitTextToSize(title, 170);
                
                // Create a proper title page
                doc.text(titleLines, 105, 80, { align: 'center' });
                
                // Author information
                doc.setFontSize(12);
                doc.setFont('helvetica', 'normal');
                doc.text('Social Media Journalism Analysis', 105, 110, { align: 'center' });
                
                // Date
                doc.setFontSize(12);
                doc.setFont('helvetica', 'italic');
                doc.text(`Generated on ${currentDate}`, 105, 120, { align: 'center' });
                
                // Add a new page for the content
                doc.addPage();
                
                // Add sections with proper academic formatting
                let yPos = 20;
                
                // Add sections
                yPos = this.addSection(doc, 'Abstract', abstract, yPos);
                yPos = this.addSection(doc, 'Introduction', introduction, yPos);
                yPos = this.addSection(doc, 'Methodology', methodology, yPos);
                yPos = this.addSection(doc, 'Results', results, yPos);
                yPos = this.addSection(doc, 'Discussion', discussion, yPos);
                yPos = this.addSection(doc, 'Conclusion', conclusion, yPos);
                yPos = this.addSection(doc, 'References', references, yPos);
                
                // Add page numbers
                const pageCount = doc.getNumberOfPages();
                for (let i = 2; i <= pageCount; i++) {
                    doc.setPage(i);
                    doc.setFontSize(10);
                    doc.setFont('helvetica', 'normal');
                    doc.text(`Page ${i - 1} of ${pageCount - 1}`, 105, 290, { align: 'center' });
                    
                    // Add header with title (abbreviated if needed)
                    const shortTitle = title.length > 50 ? title.substring(0, 50) + '...' : title;
                    doc.text(shortTitle, 105, 10, { align: 'center' });
                }
                
                // Save the PDF
                const filename = `journalism-analysis-${new Date().toISOString().split('T')[0]}.pdf`;
                doc.save(filename);
                
                // Close settings panel and show success message
                if (this.settingsPanel.classList.contains('active')) {
                    this.toggleSettings();
                }
                this.showToast('Journalism analysis exported successfully!');
            } catch (error) {
                console.error('Error generating PDF:', error);
                this.showToast('Failed to generate PDF. Please try again.');
            }
        }, 500); // Small delay to allow the UI to update
    }
    
    addSection(doc, title, content, yPos) {
        // Add a new page if we're near the bottom
        if (yPos > 250) {
            doc.addPage();
            yPos = 20;
        }
        
        // Section title with proper formatting
        doc.setFontSize(16);
        doc.setFont('helvetica', 'bold');
        doc.text(title, 20, yPos);
        yPos += 8;
        
        // Add a small line under section title
        doc.setDrawColor(70, 70, 70);
        doc.setLineWidth(0.5);
        doc.line(20, yPos, 60, yPos);
        yPos += 7;
        
        // Section content with proper line spacing
        doc.setFontSize(11);
        doc.setFont('helvetica', 'normal');
        const contentLines = doc.splitTextToSize(content, 170);
        
        // Check if content would extend beyond page, and add a new page if needed
        if (yPos + (contentLines.length * 6) > 280) {
            doc.addPage();
            yPos = 20;
        }
        
        doc.text(contentLines, 20, yPos);
        
        // Calculate new Y position with appropriate spacing between sections
        return yPos + (contentLines.length * 6) + 15;
    }
    
    generateAbstract() {
        // Extract key points from the conversation
        const userQueries = this.messages
            .filter(msg => msg.role === 'user')
            .map(msg => msg.content);
            
        const assistantResponses = this.messages
            .filter(msg => msg.role === 'assistant')
            .map(msg => msg.content);
        
        if (userQueries.length === 0 || assistantResponses.length === 0) {
            return 'No conversation data available.';
        }
        
        // Generate a journalism-focused abstract based on the conversation
        return `This analysis presents a journalistic examination of ${userQueries[0]}. 
        The investigation explores coverage across social media discussions and news sources, 
        utilizing a Retrieval-Augmented Generation (RAG) approach to synthesize information from multiple platforms. 
        Through ${userQueries.length} distinct inquiries, this investigation examines reporting trends, public discourse, 
        and media narratives on the topic. The analysis reveals how news framing and social media conversations 
        interact to shape public understanding of current events, highlighting the evolving landscape of 
        digital journalism and information consumption.`;
    }
    
    generateIntroduction() {
        const firstUserQuery = this.messages.find(msg => msg.role === 'user')?.content || '';
        
        return `The modern media landscape has fundamentally transformed how news is produced, distributed, and consumed. 
        This analysis examines reporting on "${firstUserQuery}" using a cross-platform approach 
        that integrates traditional news sources with social media content. The investigation employs a Retrieval-Augmented 
        Generation (RAG) system that enhances journalistic analysis with content retrieved 
        from Reddit, YouTube, and established news publications. This methodology provides a comprehensive framework 
        for examining both professional reporting and public discourse on current events. The analysis 
        aims to identify reporting patterns, editorial perspectives, and audience engagement across platforms, 
        while evaluating the challenges and opportunities of modern digital journalism. By examining how stories are 
        framed and discussed across different media environments, this investigation contributes to our understanding 
        of contemporary news ecosystems and information flow.`;
    }
    
    generateMethodology() {
        const hasNewsSourcesRefs = this.messages.some(msg => 
            msg.sources && msg.sources.some(src => src.type === 'news')
        );
        
        const hasSocialMediaRefs = this.messages.some(msg => 
            msg.sources && msg.sources.some(src => src.type === 'social_media')
        );
        
        return `This journalistic investigation employed a Retrieval-Augmented Generation (RAG) system that integrates 
        ${hasSocialMediaRefs ? 'user-generated content from social platforms including Reddit and YouTube' : 'various content sources'} 
        ${hasNewsSourcesRefs ? 'with reporting from established news publishers' : ''}. 
        The methodology follows standard practices in computational journalism and data-driven reporting:

        1. Story Focus Identification: Each inquiry was analyzed to determine the central journalistic angles and key information needs
        2. Source Retrieval: Relevant reporting and commentary was retrieved from indexed sources using semantic search technology
        3. Editorial Prioritization: ${hasNewsSourcesRefs ? 'Established news reporting was prioritized over user-generated content' : 'Sources were ranked by journalistic relevance and credibility'}
        4. Narrative Synthesis: A Gemini Large Language Model synthesized the information into cohesive journalistic analysis
        5. Source Attribution: All sources were properly cited following journalistic standards of transparency

        The analysis encompasses ${this.messages.filter(msg => msg.role === 'user').length} distinct journalistic inquiries and 
        ${this.messages.filter(msg => msg.role === 'assistant').length} comprehensive analyses, with citations from 
        ${this.messages.reduce((count, msg) => count + (msg.sources?.length || 0), 0)} unique sources including news publications and social media platforms.`;
    }
    
    generateResults() {
        // Extract conversation exchange summaries
        const exchanges = [];
        let currentExchange = { query: '', response: '', sources: [] };
        
        for (let i = 0; i < this.messages.length; i++) {
            const msg = this.messages[i];
            
            if (msg.role === 'user') {
                if (currentExchange.query) {
                    exchanges.push({...currentExchange});
                }
                currentExchange = { 
                    query: msg.content,
                    response: '',
                    sources: []
                };
            } else if (msg.role === 'assistant') {
                currentExchange.response = msg.content;
                currentExchange.sources = msg.sources || [];
            }
        }
        
        // Add the last exchange if present
        if (currentExchange.query && currentExchange.response) {
            exchanges.push(currentExchange);
        }
        
        // Generate a summary of the results with journalistic focus
        let result = `This investigation yielded ${exchanges.length} distinct analytical segments on the topic. `;
        
        if (exchanges.length > 0) {
            // Summarize key findings with journalism focus
            result += `Key journalistic findings include:\n\n`;
            
            exchanges.forEach((exchange, index) => {
                const newsSourceCount = exchange.sources.filter(s => s.type === 'news').length;
                const socialSourceCount = exchange.sources.filter(s => s.type === 'social_media').length;
                
                result += `Segment ${index + 1}: The inquiry into "${exchange.query.substring(0, 50)}${exchange.query.length > 50 ? '...' : ''}" `;
                result += `incorporated analysis of ${newsSourceCount} news reports and ${socialSourceCount} social media discussions. `;
                
                if (index < 2 && exchanges.length > 3) {  // Add details for first couple exchanges only to save space
                    if (newsSourceCount > 0) {
                        const newsPublishers = [...new Set(exchange.sources
                            .filter(s => s.type === 'news' && s.publisher)
                            .map(s => s.publisher))];
                            
                        if (newsPublishers.length > 0) {
                            result += `Coverage was drawn from ${newsPublishers.join(', ')}, `;
                            result += `providing professional journalistic perspectives. `;
                        }
                    }
                    
                    if (socialSourceCount > 0) {
                        const platforms = [...new Set(exchange.sources
                            .filter(s => s.type === 'social_media')
                            .map(s => s.platform))];
                            
                        if (platforms.length > 0) {
                            result += `Public discourse was analyzed from ${platforms.join(', ')}, `;
                            result += `offering insights into audience reception and discussion. `;
                        }
                    }
                }
                
                result += '\n\n';
            });
        }
        
        return result;
    }
    
    generateDiscussion() {
        // Count the types of sources used
        const newsSourcesCount = this.messages.reduce((count, msg) => 
            count + (msg.sources?.filter(s => s.type === 'news').length || 0), 0);
            
        const socialSourcesCount = this.messages.reduce((count, msg) => 
            count + (msg.sources?.filter(s => s.type === 'social_media').length || 0), 0);
        
        // Extract unique platforms
        const platforms = [...new Set(this.messages.flatMap(msg => 
            (msg.sources || [])
                .filter(s => s.type === 'social_media')
                .map(s => s.platform)
        ))];
        
        // Extract unique publishers
        const publishers = [...new Set(this.messages.flatMap(msg => 
            (msg.sources || [])
                .filter(s => s.type === 'news' && s.publisher)
                .map(s => s.publisher)
        ))];
        
        return `This cross-platform media analysis demonstrates the evolving relationship between traditional journalism 
        and digital discourse in the modern information ecosystem. Throughout this investigation, ${newsSourcesCount} news articles and 
        ${socialSourcesCount} social media contributions were examined to provide a comprehensive view of the media landscape 
        surrounding this topic.

        The integration of user perspectives from ${platforms.join(', ')} offered insight into public reception and 
        audience engagement with news coverage, while reporting from outlets such as ${publishers.slice(0, 3).join(', ')} 
        provided professional journalistic context and factual frameworks. This multi-platform approach reveals how 
        modern journalism operates within a complex information environment where traditional and social media constantly interact.

        A significant observation is the distinct framing differences between professional reporting and public discussion, 
        highlighting the growing importance of media literacy in distinguishing between vetted journalism and user-generated content. 
        The analysis also reveals how audience feedback on social platforms increasingly influences subsequent news coverage, 
        creating a dynamic feedback loop in the contemporary news cycle.

        The investigation further identifies variations in journalistic quality and editorial standards across sources, 
        underscoring the challenges facing news consumers in an era of information abundance. This finding emphasizes 
        the critical role of source evaluation and editorial transparency in modern journalism practice.`;
    }
    
    generateConclusion() {
        return `This analysis demonstrates the effectiveness of computational journalism approaches in synthesizing 
        information across traditional and social media platforms to provide comprehensive coverage of contemporary issues. 
        The investigation illustrates how integrating professional reporting with audience discourse creates a more 
        complete picture of the modern media ecosystem and news reception.

        Several key journalistic implications emerge from this analysis:

        1. Cross-platform media analysis provides critical context that single-source journalism often lacks
        2. The relationship between news reporting and social media discourse reveals important insights about information flow and audience engagement
        3. Source attribution and transparency remain fundamental to maintaining journalistic credibility in the digital age
        4. Editorial judgment in source selection and prioritization significantly impacts the resulting narrative

        As journalism continues to evolve in the digital environment, these findings suggest opportunities for more 
        integrated approaches to reporting that acknowledge and incorporate audience perspectives while maintaining 
        professional standards. Future developments in computational journalism could further enhance these capabilities 
        by improving source verification methods, detecting misinformation earlier in the news cycle, and providing 
        more nuanced analysis of how news framing shapes public discourse.`;
    }
    
    generateReferences() {
        // Collect all unique sources across messages
        const allSources = this.messages.flatMap(msg => msg.sources || []);
        
        // Separate by type and deduplicate (using URL as the key for news, and a combination of factors for social)
        const newsSourcesMap = new Map();
        const socialSourcesMap = new Map();
        
        allSources.forEach(source => {
            if (source.type === 'news' && source.url) {
                if (!newsSourcesMap.has(source.url)) {
                    newsSourcesMap.set(source.url, source);
                }
            } else if (source.type === 'social_media') {
                const key = `${source.platform}-${source.username}-${source.content_type}`;
                if (!socialSourcesMap.has(key)) {
                    socialSourcesMap.set(key, source);
                }
            }
        });
        
        // Convert to arrays and sort
        const newsSources = Array.from(newsSourcesMap.values())
            .sort((a, b) => (a.publisher || '').localeCompare(b.publisher || ''));
            
        const socialSources = Array.from(socialSourcesMap.values())
            .sort((a, b) => (a.platform || '').localeCompare(b.platform || ''));
        
        // Format references in journalistic style
        let references = '';
        
        if (newsSources.length > 0) {
            references += 'News Articles:\n\n';
            
            newsSources.forEach((source, index) => {
                const publishDate = source.publication_date ? 
                    new Date(source.publication_date).toLocaleDateString('en-US', {
                        year: 'numeric', 
                        month: 'short', 
                        day: 'numeric'
                    }) : 'n.d.';
                    
                references += `[${index + 1}] ${source.publisher || 'Unknown Publisher'}. (${publishDate}). "${source.title}". `;
                references += `Retrieved from ${source.url}\n\n`;
            });
        }
        
        if (socialSources.length > 0) {
            references += 'Social Media Sources:\n\n';
            
            socialSources.forEach((source, index) => {
                const date = new Date().toLocaleDateString('en-US', {
                    year: 'numeric', 
                    month: 'short', 
                    day: 'numeric'
                });
                
                references += `[${newsSources.length + index + 1}] ${source.username} (${date}). `;
                references += `${source.content_type} post on ${source.platform}. `;
                references += `${source.url ? `Retrieved from ${source.url}` : 'Content accessed through social media API'}\n\n`;
            });
        }
        
        if (references === '') {
            references = 'No references available.';
        }
        
        return references;
    }
}

// Example query functionality
function sendExample(element) {
    const query = element.textContent.replace(/['"]/g, '');
    const messageInput = document.getElementById('messageInput');
    messageInput.value = query;
    
    // Trigger send
    const app = window.chatApp;
    if (app) {
        app.sendMessage();
    }
}

// Global settings toggle
function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    panel.classList.toggle('open');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});

// Handle page visibility for better UX
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.chatApp) {
        // Refresh connection status when page becomes visible
        setTimeout(() => {
            if (!window.chatApp.isInitialized) {
                window.chatApp.initializeSystem();
            }
        }, 1000);
    }
});