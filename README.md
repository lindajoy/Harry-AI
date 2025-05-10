# Harry-AI Project

## Overview

Harry-AI is a project bringing Harry Potter content to life through modern AI technology. The project consists of two main applications:

1. **Jinx**: A backend RAG (Retrieval-Augmented Generation) system that delivers Harry Potter content in Gen Z language.
2. **Arcane**: A frontend Angular application that provides a user-friendly chat interface to interact with the system.

## Jinx - RAG Backend

Jinx uses a Retrieval-Augmented Generation system powered by Google's Gemini models to provide context-aware responses based on Harry Potter content. The content has been modernized with Gen Z language and slang.

### Key Features

- Document processing and chunking for efficient retrieval
- Vector embeddings using Google Generative AI models
- Contextual responses based on relevant text chunks
- Gen Z language styling for all content

### Requirements

### Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up your Google API credentials in a `.env` file or as environment variables
3. Run the indexing pipeline to create vector embeddings of the content
4. Start the API server: `uvicorn server.websocket_server:app --reload`

## Arcane - Angular Frontend

Arcane provides a modern, responsive user interface for interacting with the Harry Potter content.

### Key Features

- Clean, intuitive chat interface
- Customizable tone selection
- Support for including sources in responses
- Dark/light mode support

### Development

This project was generated with Angular CLI version 17.3.9.

#### Requirements

- Node.js and npm
- Angular CLI

#### Running the App

1. Install dependencies: `npm install`
2. Start development server: `ng serve`
3. Navigate to `http://localhost:4200/`

#### Building

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory.

## Content

The project includes modernized versions of Harry Potter chapters. For example, Chapter 3 "The Letters from ur mom" contains the story of Harry receiving his Hogwarts letters, but written in modern Gen Z slang and references.

## Usage

Users can ask questions about Harry Potter through the Arcane interface, and the system will retrieve relevant passages from the modernized content, using the Jinx RAG system to generate contextual, Gen Z-styled responses.

Example query: "What happened when Harry got his first Hogwarts letter?"
