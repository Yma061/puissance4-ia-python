import customtkinter as ctk
import math

from game.board import *
from game.rules import winning_move
from game.ai_minimax import minimax
from game.ai_deep import predict_move


PLAYER_PIECE = 1
AI_PIECE = 2


# =====================================================
# MENU PRINCIPAL
# =====================================================

class MainMenu(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Puissance 4")
        self.geometry("500x500")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.create_widgets()

    def create_widgets(self):

        title = ctk.CTkLabel(
            self,
            text="🎮 PUISSANCE 4",
            font=("Arial", 32, "bold")
        )
        title.pack(pady=40)

        ctk.CTkButton(
            self,
            text="👥 Jouer en 1 vs 1",
            height=50,
            command=self.start_1v1
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text="🤖 Jouer contre IA Minimax",
            height=50,
            command=self.start_minimax
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text="🧠 Jouer contre IA Deep Learning",
            height=50,
            command=self.start_deep
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text="❌ Quitter",
            height=50,
            fg_color="red",
            command=self.destroy
        ).pack(pady=20)

    def start_1v1(self):
        self.destroy()
        Connect4UI(mode="1v1").mainloop()

    def start_minimax(self):
        self.destroy()
        Connect4UI(mode="minimax").mainloop()

    def start_deep(self):
        self.destroy()
        Connect4UI(mode="deep").mainloop()


# =====================================================
# FENÊTRE DE JEU
# =====================================================

class Connect4UI(ctk.CTk):

    def __init__(self, mode="minimax"):
        super().__init__()

        self.title("Puissance 4 - Jeu")
        self.geometry("720x720")

        self.mode = mode
        self.board = create_board()
        self.turn = 0
        self.game_over = False

        self.cells = []

        self.create_widgets()
        self.draw_board()

    # -------------------------------------------------

    def create_widgets(self):

        # Boutons colonnes
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=10)

        for col in range(COLUMN_COUNT):
            ctk.CTkButton(
                top_frame,
                text=str(col),
                width=70,
                command=lambda c=col: self.play_turn(c)
            ).grid(row=0, column=col, padx=5)

        # Grille visuelle
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(pady=20)

        for r in range(ROW_COUNT):
            row_cells = []
            for c in range(COLUMN_COUNT):
                cell = ctk.CTkLabel(
                    self.grid_frame,
                    text="",
                    width=70,
                    height=70,
                    corner_radius=35,
                    fg_color="gray"
                )
                cell.grid(row=r, column=c, padx=5, pady=5)
                row_cells.append(cell)
            self.cells.append(row_cells)

    # -------------------------------------------------

    def draw_board(self):

        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):

                piece = self.board[r][c]

                if piece == PLAYER_PIECE:
                    color = "red"
                elif piece == AI_PIECE:
                    color = "yellow"
                else:
                    color = "gray"

                self.cells[ROW_COUNT - 1 - r][c].configure(fg_color=color)

    # -------------------------------------------------

    def play_turn(self, col):

        if self.game_over:
            return

        if not is_valid_location(self.board, col):
            return

        row = get_next_open_row(self.board, col)

        # ------------------ 1v1 ------------------
        if self.mode == "1v1":

            piece = PLAYER_PIECE if self.turn == 0 else AI_PIECE
            drop_piece(self.board, row, col, piece)

            if winning_move(self.board, piece):
                self.draw_board()
                winner = "🔴 Joueur 1 gagne !" if piece == 1 else "🟡 Joueur 2 gagne !"
                self.show_winner(winner)
                self.game_over = True
                return

            self.turn = 1 - self.turn
            self.draw_board()

        # ------------------ VS IA ------------------
        else:

            # Coup joueur
            drop_piece(self.board, row, col, PLAYER_PIECE)

            if winning_move(self.board, PLAYER_PIECE):
                self.draw_board()
                self.show_winner("🎉 Tu as gagné !")
                self.game_over = True
                return

            self.draw_board()

            # Laisser petit délai pour IA
            self.after(300, self.ai_move)

    # -------------------------------------------------

    def ai_move(self):

        if self.game_over:
            return

        # Choix IA selon mode
        if self.mode == "minimax":
            col, _ = minimax(self.board, 4, -math.inf, math.inf, True)

        elif self.mode == "deep":
            col = predict_move(self.board)

        else:
            return

        if not is_valid_location(self.board, col):
            return

        row = get_next_open_row(self.board, col)
        drop_piece(self.board, row, col, AI_PIECE)

        if winning_move(self.board, AI_PIECE):
            self.draw_board()
            self.show_winner("🤖 L'IA a gagné !")
            self.game_over = True
            return

        self.draw_board()

    # -------------------------------------------------

    def show_winner(self, message):

        popup = ctk.CTkToplevel(self)
        popup.geometry("300x150")
        popup.title("Fin de partie")

        ctk.CTkLabel(
            popup,
            text=message,
            font=("Arial", 18)
        ).pack(pady=20)

        ctk.CTkButton(
            popup,
            text="Retour Menu",
            command=self.return_menu
        ).pack(pady=10)

    # -------------------------------------------------

    def return_menu(self):
        self.destroy()
        MainMenu().mainloop()